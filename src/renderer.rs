use crate::{
    camera::{Camera, Projection},
    cmap::ColorMapGPU,
    volume::{Aabb, Volume, VolumeGPU},
};

use cgmath::{EuclideanSpace, Matrix4, SquareMatrix, Vector3, Vector4, Zero};
use wgpu::util::DeviceExt;

pub struct VolumeRenderer {
    pipeline: wgpu::RenderPipeline,
    sampler_nearest: wgpu::Sampler,
    sampler_linear: wgpu::Sampler,
    format: wgpu::TextureFormat,
}

impl VolumeRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[
                &Self::bind_group_layout(device),
                &ColorMapGPU::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/raymarch.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None, //Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,

            ..Default::default()
        });
        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        VolumeRenderer {
            pipeline,
            sampler_nearest,
            sampler_linear,
            format: color_format,
        }
    }

    pub fn prepare<'a, P: Projection>(
        &mut self,
        device: &wgpu::Device,
        volume: &VolumeGPU,
        camera: &Camera<P>,
        render_settings: &RenderSettings,
        cmap: &'a ColorMapGPU,
    ) -> PerFrameData<'a> {
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::from(camera)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("settnigs buffer"),
            contents: bytemuck::bytes_of(&RenderSettingsUniform::from_settings(
                &render_settings,
                &volume.volume,
            )),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let step = ((volume.volume.timesteps() - 1) as f32 * render_settings.time) as usize;
        // TODO maybe create all bindgroups once and not on the fly per frame
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume renderer bind group"),
            layout: &Self::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &volume.textures[step].create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &volume.textures[(step + 1) % volume.volume.timesteps() as usize]
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(
                        if render_settings.spatial_filter == wgpu::FilterMode::Nearest {
                            &self.sampler_nearest
                        } else {
                            &self.sampler_linear
                        },
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        camera_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        settings_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });
        PerFrameData {
            bind_group,
            cmap_bind_group: cmap.bindgroup(),
        }
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        frame_data: &'rpass PerFrameData,
    ) {
        render_pass.set_bind_group(0, &frame_data.bind_group, &[]);
        render_pass.set_bind_group(1, frame_data.cmap_bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw(0..4, 0..1);
    }

    pub(crate) fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume renderer bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
}

pub struct PerFrameData<'a> {
    pub(crate) bind_group: wgpu::BindGroup,
    cmap_bind_group: &'a wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// the cameras view matrix
    pub(crate) view_matrix: Matrix4<f32>,
    /// inverse view matrix
    pub(crate) view_inv_matrix: Matrix4<f32>,

    // the cameras projection matrix
    pub(crate) proj_matrix: Matrix4<f32>,

    // inverse projection matrix
    pub(crate) proj_inv_matrix: Matrix4<f32>,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_matrix: Matrix4::identity(),
            view_inv_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
            proj_inv_matrix: Matrix4::identity(),
        }
    }
}

impl CameraUniform {
    pub(crate) fn set_view_mat(&mut self, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;
        self.view_inv_matrix = view_matrix.invert().unwrap();
    }

    pub(crate) fn set_proj_mat(&mut self, proj_matrix: Matrix4<f32>) {
        self.proj_matrix = proj_matrix;
        self.proj_inv_matrix = proj_matrix.invert().unwrap();
    }

    pub fn set_camera(&mut self, camera: &Camera<impl Projection>) {
        self.set_proj_mat(camera.proj_matrix());
        self.set_view_mat(camera.view_matrix());
    }
}

impl<P: Projection> From<&Camera<P>> for CameraUniform {
    fn from(camera: &Camera<P>) -> Self {
        let mut uniform = CameraUniform::default();
        uniform.set_camera(camera);
        uniform
    }
}
#[derive(Debug, Clone)]
pub struct RenderSettings {
    pub clipping_aabb: Option<Aabb<f32>>,
    pub time: f32,
    pub step_size: f32,
    pub spatial_filter: wgpu::FilterMode,
    pub temporal_filter: wgpu::FilterMode,
    pub distance_scale: f32,
    pub vmin: Option<f32>,
    pub vmax: Option<f32>,
    pub gamma_correction: bool,

    pub render_volume: bool,
    pub render_iso: bool,
    pub use_cube_surface_grad: bool,
    pub iso_shininess: f32,
    pub iso_threshold: f32,

    pub iso_ambient_color: Vector3<f32>,
    pub iso_specular_color: Vector3<f32>,
    pub iso_light_color: Vector3<f32>,
    pub iso_diffuse_color: Vector4<f32>,

    pub ssao: bool,
    pub ssao_radius: f32,
    pub ssao_bias: f32,
    pub ssao_kernel_size: u32,

    pub background_color: wgpu::Color,
}

impl std::hash::Hash for RenderSettings {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.clipping_aabb.hash(state);
        self.time.to_bits().hash(state);
        self.step_size.to_bits().hash(state);
        self.spatial_filter.hash(state);
        self.temporal_filter.hash(state);
        self.distance_scale.to_bits().hash(state);
        self.vmin.map(|v| v.to_bits()).hash(state);
        self.vmax.map(|v| v.to_bits()).hash(state);
        self.gamma_correction.hash(state);
        self.render_volume.hash(state);
        self.render_iso.hash(state);
        self.use_cube_surface_grad.hash(state);
        self.iso_shininess.to_bits().hash(state);
        self.iso_threshold.to_bits().hash(state);

        self.iso_ambient_color.x.to_bits().hash(state);
        self.iso_ambient_color.y.to_bits().hash(state);
        self.iso_ambient_color.z.to_bits().hash(state);

        self.iso_specular_color.x.to_bits().hash(state);
        self.iso_specular_color.y.to_bits().hash(state);
        self.iso_specular_color.z.to_bits().hash(state);

        self.iso_light_color.x.to_bits().hash(state);
        self.iso_light_color.y.to_bits().hash(state);
        self.iso_light_color.z.to_bits().hash(state);

        self.iso_diffuse_color.x.to_bits().hash(state);
        self.iso_diffuse_color.y.to_bits().hash(state);
        self.iso_diffuse_color.z.to_bits().hash(state);
        self.iso_diffuse_color.w.to_bits().hash(state);

        self.ssao.hash(state);
        self.ssao_radius.to_bits().hash(state);
        self.ssao_bias.to_bits().hash(state);
        self.ssao_kernel_size.hash(state);
        self.background_color.r.to_bits().hash(state);
        self.background_color.g.to_bits().hash(state);
        self.background_color.b.to_bits().hash(state);
        self.background_color.a.to_bits().hash(state);
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            clipping_aabb: None,
            time: 0.,
            step_size: 1e-4,
            spatial_filter: wgpu::FilterMode::Linear,
            temporal_filter: wgpu::FilterMode::Linear,
            distance_scale: 1.,
            vmin: None,
            vmax: None,
            gamma_correction: false,
            render_volume: true,
            render_iso: false,
            use_cube_surface_grad: false,
            iso_shininess: 20.0,
            iso_threshold: 0.5,
            iso_ambient_color: Vector3::zero(),
            iso_specular_color: Vector3::new(0.7, 0.7, 0.7),
            iso_light_color: Vector3::new(1., 1., 1.),
            iso_diffuse_color: Vector4::new(1.0, 0.871, 0.671, 1.0),
            ssao: true,
            ssao_radius: 0.4,
            ssao_bias: 0.02,
            ssao_kernel_size: 64,
            background_color: wgpu::Color::BLACK,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderSettingsUniform {
    volume_aabb_min: Vector4<f32>,
    volume_aabb_max: Vector4<f32>,
    clipping_min: Vector4<f32>,
    clipping_max: Vector4<f32>,

    time: f32,
    time_steps: u32,
    temporal_filter: u32,
    spatial_filter: u32,

    distance_scale: f32,
    vmin: f32,
    vmax: f32,
    gamma_correction: u32,

    iso_ambient_color: Vector4<f32>,
    iso_specular_color: Vector4<f32>,
    iso_light_color: Vector4<f32>,
    iso_diffuse_color: Vector4<f32>,

    render_volume: u32,
    render_iso: u32,
    use_cube_surface_grad: u32,
    iso_shininess: f32,

    iso_threshold: f32,
    step_size: f32,
    ssao_radius: f32,
    ssao_bias: f32,

    background_color: Vector4<f32>,

    ssao_kernel_size: u32,
    _pad: [u32; 3],
}
impl RenderSettingsUniform {
    pub fn from_settings(settings: &RenderSettings, volume: &Volume) -> Self {
        let volume_aabb = volume.aabb;

        Self {
            volume_aabb_min: volume_aabb.min.to_vec().extend(0.),
            volume_aabb_max: volume_aabb.max.to_vec().extend(0.),
            time: settings.time,
            time_steps: volume.timesteps() as u32,
            clipping_min: settings
                .clipping_aabb
                .map(|bb| bb.min.to_vec().extend(0.))
                .unwrap_or(RenderSettingsUniform::default().clipping_min),
            clipping_max: settings
                .clipping_aabb
                .map(|bb| bb.max.to_vec().extend(0.))
                .unwrap_or(RenderSettingsUniform::default().clipping_max),
            step_size: settings.step_size,
            temporal_filter: settings.temporal_filter as u32,
            spatial_filter: settings.spatial_filter as u32,
            distance_scale: settings.distance_scale,
            vmin: settings.vmin.unwrap_or(volume.min_value),
            vmax: settings.vmax.unwrap_or(volume.max_value),
            gamma_correction: settings.gamma_correction as u32,
            render_volume: settings.render_volume as u32,
            render_iso: settings.render_iso as u32,
            use_cube_surface_grad: settings.use_cube_surface_grad as u32,
            iso_shininess: settings.iso_shininess,
            iso_threshold: settings.iso_threshold,
            iso_ambient_color: settings.iso_ambient_color.extend(0.),
            iso_specular_color: settings.iso_specular_color.extend(0.),
            iso_light_color: settings.iso_light_color.extend(0.),
            iso_diffuse_color: settings.iso_diffuse_color,
            ssao_radius: settings.ssao_radius,
            ssao_bias: settings.ssao_bias,
            ssao_kernel_size: settings.ssao_kernel_size,
            background_color: Vector4::new(
                settings.background_color.r as f32,
                settings.background_color.g as f32,
                settings.background_color.b as f32,
                settings.background_color.a as f32,
            ),
            _pad: [0; 3],
        }
    }
}

impl Default for RenderSettingsUniform {
    fn default() -> Self {
        Self {
            volume_aabb_min: Vector4::new(-1., -1., -1., 0.),
            volume_aabb_max: Vector4::new(1., 1., 1., 0.),
            clipping_min: Vector4::zero(),
            clipping_max: Vector4::new(1., 1., 1., 0.),
            time: 0.,
            time_steps: 1,
            step_size: 0.01,
            temporal_filter: wgpu::FilterMode::Linear as u32,
            spatial_filter: wgpu::FilterMode::Nearest as u32,
            distance_scale: 1.,
            vmin: 0.,
            vmax: 1.,
            gamma_correction: 0,
            render_volume: 1,
            render_iso: 0,
            use_cube_surface_grad: 0,
            iso_shininess: 20.0,
            iso_threshold: 0.5,
            iso_ambient_color: Vector4::zero(),

            iso_specular_color: Vector4::new(1., 1., 1., 0.),
            iso_light_color: Vector4::new(1., 1., 1., 0.),
            iso_diffuse_color: Vector4::new(1.0, 0.871, 0.671, 1.0),

            ssao_radius: 0.4,
            ssao_bias: 0.02,
            ssao_kernel_size: 64,
            background_color: Vector4::new(0., 0., 0., 1.),
            _pad: [0; 3],
        }
    }
}
