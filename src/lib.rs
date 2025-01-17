use camera::{Camera, OrthographicProjection};
use controller::CameraController;
use egui::FullOutput;
use image::ImageReader;
use renderer::{Display, FrameBuffer, IsoSettings, RenderSettings, RenderState, VolumeRenderer};
use std::{collections::HashMap, io::Cursor, path::PathBuf, sync::Arc};
use volume::VolumeGPU;

#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use wgpu::Backends;

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;

use cgmath::Vector2;
use winit::{
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use crate::volume::Volume;

pub mod camera;
pub mod cmap;
mod controller;
pub mod offline;
#[cfg(feature = "python")]
pub mod py;
pub mod renderer;
mod ssao;
mod ui;
mod ui_renderer;
mod viewer;
pub mod volume;
pub use viewer::viewer;
mod presets;

use presets::{Preset, PRESETS};

#[derive(Debug)]
pub struct RenderConfig {
    pub no_vsync: bool,
    pub show_colormap_editor: bool,
    pub show_volume_info: bool,
    #[cfg(feature = "colormaps")]
    pub show_cmap_select: bool,
    pub duration: Option<Duration>,
    pub preset: Option<Preset>,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();

        let required_features = wgpu::Features::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    required_limits: adapter.limits(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device,
            queue,
            adapter,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    controller: CameraController,
    camera: Camera<OrthographicProjection>,
    ui_renderer: ui_renderer::EguiWGPU,
    ui_visible: bool,

    volume: VolumeGPU,
    renderer: VolumeRenderer,

    render_state: RenderState,

    colormap_editor_visible: bool,
    volume_info_visible: bool,
    #[cfg(feature = "colormaps")]
    cmap_select_visible: bool,

    ssao: ssao::SSAO,
    display: Display,

    frame_buffer: FrameBuffer,

    show_box: bool,

    presets: HashMap<String, Preset>,
    selected_preset: Option<String>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new(
        window: Window,
        volume: Volume,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size().to_logical(window.scale_factor());
        if size.width == 0 || size.height == 0 {
            size = LogicalSize::new(800, 600);
        }
        let window = Arc::new(window);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::all().symmetric_difference(Backends::BROWSER_WEBGPU),
            ..Default::default()
        });

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let max_size = device.limits().max_texture_dimension_2d;
        window.set_max_inner_size(Some(PhysicalSize::new(max_size, max_size)));

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();
        let surface_format = surface_format;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let render_format = surface_format;
        let renderer = VolumeRenderer::new(device, render_format);

        let mut presets = PRESETS.clone();

        #[cfg(target_arch = "wasm32")]
        {
            match crate::presets::Presets::from_local_storage() {
                Ok(local_presets) => {
                    log::info!(
                        "loaded {} presets from local storage",
                        local_presets.0.len()
                    );
                    presets.extend(local_presets.0);
                }
                Err(err) => log::error!("failed to load presets from local storage: {}", err),
            }
        }

        let mut selected_preset = None;
        let render_settings = render_config
            .preset
            .as_ref()
            .map(|preset| {
                presets.insert(preset.name.clone(), preset.clone());
                selected_preset = Some(preset.name.clone());
                preset.render_settings.clone()
            })
            .unwrap_or(RenderSettings {
                iso_surface: IsoSettings {
                    threshold: (volume.min_value(0) + volume.max_value(0)) / 2.,
                    color_channel: 1.min(volume.channels()),
                    ..Default::default()
                },
                ..Default::default()
            });

        let render_state = RenderState {
            settings: render_settings,
            gamma_correction: !render_format.is_srgb(),
            step_size: 2e-3,
        };

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = volume.aabb.center();

        let radius = volume.aabb.radius();
        let ratio = size.width as f32 / size.height as f32;
        let camera = Camera::new_aabb_iso(
            volume.aabb.clone(),
            OrthographicProjection::new(Vector2::new(ratio, 1.) * 2. * radius, 1e-4, 100.),
            None,
        );

        let volumes_gpu = VolumeGPU::new(device, queue, volume);

        let ssao = ssao::SSAO::new(device);

        let frame_buffer = FrameBuffer::new(
            device,
            Vector2::new(config.width, config.height),
            render_format,
        );

        let display = Display::new(device, surface_format);

        let mut me = Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            controller,
            ui_renderer,
            ui_visible: true,
            camera,

            volume: volumes_gpu,
            renderer,

            render_state,
            colormap_editor_visible: render_config.show_colormap_editor,
            volume_info_visible: render_config.show_volume_info,
            #[cfg(feature = "colormaps")]
            cmap_select_visible: render_config.show_cmap_select,
            ssao,
            show_box: false,
            presets,
            selected_preset,
            frame_buffer,
            display,
        };

        if let Some(preset) = render_config.preset.as_ref() {
            me.set_preset(preset.clone());
        }

        Ok(me)
    }

    pub(crate) fn set_preset(&mut self, preset: Preset) {
        self.render_state.settings = preset.render_settings;
        self.selected_preset = Some(preset.name);
        if let Some(camera) = preset.camera {
            let aabb = self.volume.volume.aabb.clone();
            self.controller.center = aabb.center();

            self.camera = Camera::new_aabb_iso(aabb, self.camera.projection.clone(), Some(camera));
            self.controller.reset();
        }
    }

    fn load_file(&mut self, path: &PathBuf) -> anyhow::Result<()> {
        let reader = std::fs::File::open(path)?;
        let volume = Volume::load(reader)?;
        let volume_gpu =
            VolumeGPU::new(&self.wgpu_context.device, &self.wgpu_context.queue, volume);
        self.volume = volume_gpu;
        // self.controller.center = volume.aabb.center();
        self.camera
            .projection
            .resize(self.config.width, self.config.height);
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            let new_width = new_size.width;
            let new_height = new_size.height;
            self.config.width = new_width;
            self.config.height = new_height;
            self.camera.projection.resize(new_width, new_height);
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.frame_buffer = FrameBuffer::new(
                &self.wgpu_context.device,
                Vector2::new(new_width, new_height),
                self.frame_buffer.color_format,
            );
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn update(&mut self, dt: Duration) -> bool {
        let old_settings = self.render_state.clone();
        let old_camera = self.camera.clone();
        self.controller.update_camera(&mut self.camera, dt);
        if !self.camera.visible(self.volume.volume.aabb) {
            self.camera = old_camera;
        }

        let volume_aabb = self.volume.volume.aabb;

        self.camera.fit_near_far(&volume_aabb);

        let request_redraw = old_settings != self.render_state || old_camera != self.camera;
        return request_redraw;
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let request_redraw = ui::ui(self);

        let shapes = self.ui_renderer.end_frame(&self.window);

        return (request_redraw, shapes);
    }

    fn render(
        &mut self,
        window: Arc<Window>,
        shapes: Option<FullOutput>,
    ) -> Result<(), wgpu::SurfaceError> {
        let window_size = self.window.inner_size();
        if window_size.width != self.config.width || window_size.height != self.config.height {
            self.resize(window_size, None);
        }

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format),
            ..Default::default()
        });

        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &mut encoder,
                shapes,
            )
        });

        let camera = self.camera.clone();
        let frame_data = self.renderer.prepare(
            &self.wgpu_context.device,
            &self.wgpu_context.queue,
            &self.volume,
            &camera,
            &self.render_state,
        );

        self.renderer
            .render(&mut encoder, &frame_data, &self.frame_buffer);

        if self.render_state.settings.iso_surface.enabled && self.render_state.settings.ssao.enabled
        {
            self.ssao.render(
                &mut encoder,
                &self.wgpu_context.device,
                &self.frame_buffer,
                &self.camera,
                &frame_data,
            );
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_rgb,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.display
                .render(&mut render_pass, &self.frame_buffer, &frame_data);
            if let Some(state) = &ui_state {
                self.ui_renderer.render(&mut render_pass, state);
            }
        }
        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.wgpu_context
            .queue
            .submit(std::iter::once(encoder.finish()));
        window.pre_present_notify();
        output.present();
        Ok(())
    }
}

pub async fn open_window(window_builder: WindowBuilder, volume: Volume, config: RenderConfig) {
    let event_loop = EventLoop::new().unwrap();

    let version = env!("CARGO_PKG_VERSION");
    let name = env!("CARGO_PKG_NAME");

    let icon = ImageReader::new(Cursor::new(include_bytes!("../public/icon.png")))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .resize(64, 64, image::imageops::FilterType::Lanczos3);
    let icon_width = icon.width();
    let icon_height = icon.height();

    let window = window_builder
        .with_title(format!("{name} {version}"))
        .with_inner_size(LogicalSize::new(800, 600))
        .with_window_icon(Some(
            winit::window::Icon::from_rgba(icon.into_rgba8().into_vec(), icon_width, icon_height)
                .unwrap(),
        ))
        .build(&event_loop)
        .unwrap();

    let min_wait = window
        .current_monitor()
        .map(|m| {
            let hz = m.refresh_rate_millihertz().unwrap_or(60_000);
            Duration::from_millis(1000000 / hz as u64)
        })
        .unwrap_or(Duration::from_millis(17));

    let mut state = WindowContext::new(window, volume, &config).await.unwrap();

    let mut last = Instant::now();

    let mut last_touch_pos: PhysicalPosition<f64> = PhysicalPosition::new(0.0, 0.0);
    event_loop.run(move |event,target|{
        match event {
            Event::NewEvents(e) =>  match e{
                winit::event::StartCause::ResumeTimeReached { .. }=>{
                    state.window.request_redraw();
                }
                _=>{}
            }
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !(state.ui_visible && state.ui_renderer.on_event(&state.window,event)) =>{
            match event {
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size, None);
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    ..
                } => {
                    state.scale_factor = *scale_factor as f32;
                }
                WindowEvent::CloseRequested => {log::info!("close!");target.exit()},
                WindowEvent::ModifiersChanged(m)=>{
                    state.controller.alt_pressed = m.state().alt_key();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(key) = event.physical_key{
                        state
                            .controller
                            .process_keyboard(key, event.state == ElementState::Pressed);
                        if key == KeyCode::KeyU && event.state == ElementState::Released{
                            state.ui_visible = !state.ui_visible;
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                        state.controller.process_scroll(*dy )
                    }
                    winit::event::MouseScrollDelta::PixelDelta(p) => {
                        state.controller.process_scroll(p.y as f32 / 100.)
                    }
                },
                WindowEvent::MouseInput { state:button_state, button, .. }=>{
                    match button {
                        winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                        winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                        _=>{}
                    }
                },
                WindowEvent::Touch(t) => {
                    if t.phase == winit::event::TouchPhase::Moved{
                        state.controller.process_mouse((t.location.x - last_touch_pos.x) as f32, (t.location.y - last_touch_pos.y) as f32);
                        last_touch_pos = t.location;
                    }else if t.phase == winit::event::TouchPhase::Started{
                        last_touch_pos = t.location;
                    }
                }
                WindowEvent::DroppedFile(file) => {
                    if let Err(e) = state.load_file(file){
                        log::error!("failed to load file: {:?}", e)
                    }
                }
                WindowEvent::RedrawRequested => {
                    if !config.no_vsync{
                        // make sure the next redraw is called with a small delay
                        target.set_control_flow(ControlFlow::wait_duration(min_wait));
                    }
                    let now = Instant::now();
                    let dt = now-last;
                    last = now;
                    let request_redraw = state.update(dt);

                    let (redraw_ui,shapes) = state.ui();

                    // check whether we need to redraw
                    if request_redraw || redraw_ui{
                        match state.render(state.window.clone(),state.ui_visible.then_some(shapes)) {
                            Ok(_) => {}
                            // Reconfigure the surface if lost
                            Err(wgpu::SurfaceError::Lost) =>{
                                log::error!("lost surface!");
                                state.resize(state.window.inner_size(), None)

                                },
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) =>target.exit(),
                            // All other errors (Outdated, Timeout) should be resolved by the next frame
                            Err(e) => println!("error: {:?}", e),
                        }
                    }
                    if config.no_vsync{
                        state.window.request_redraw();
                    }

                }
            _ => {}
        }},
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
        }
        Event::AboutToWait => {
            #[cfg(target_arch = "wasm32")]
            use winit::platform::web::WindowExtWebSys;
            #[cfg(target_arch = "wasm32")]
            if let Some(canvas) = state.window.canvas() {
                if canvas.parent_node().is_none() {
                    // The canvas has been removed from the DOM, we should exit
                    target.exit();
                    return;
                }
            }
        }
        _ => {},
}}).unwrap();
    log::info!("exit!");
}
