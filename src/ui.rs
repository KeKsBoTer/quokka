use std::{f32::consts::PI, ops::RangeInclusive, time::Duration};

use bytemuck::Zeroable;
use egui::{emath::Numeric, vec2};
use egui_plot::{Plot, PlotImage, PlotPoint};

use crate::{
    cmap::{ColorMap, COLORMAP_RESOLUTION}, presets::Preset, WindowContext
};

#[cfg(target_arch = "wasm32")]
use crate::local_storage;

#[cfg(feature = "colormaps")]
use crate::cmap::COLORMAPS;

/// returns true if a repaint is requested
pub(crate) fn ui(state: &mut WindowContext) -> bool {
    let ctx = state.ui_renderer.winit.egui_ctx();
    let with_animation = state.volume.volume.timesteps() > 1;
    egui::Window::new("Render Settings").fade_in(true).show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                if with_animation {
                    ui.label("Animation");
                    // ui.end_row();
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(&mut state.render_state.time, (0.)..=(1.))
                                .show_value(false)
                                .clamp_to_range(true)
                                .trailing_fill(true)
                                .handle_shape(style::HandleShape::Rect { aspect_ratio: 0.5 }),
                        );
                        if ui.button(if state.playing { "||" } else { "▶" }).clicked() {
                            state.playing = !state.playing;
                        }
                        ui.add(
                            egui::DragValue::from_get_set(|v| {
                                if let Some(v) = v {
                                    state.animation_duration = Duration::from_secs_f64(v);
                                    return v;
                                } else {
                                    return state.animation_duration.as_secs_f64();
                                }
                            })
                            .suffix("s")
                            .range((0.)..=1000.),
                        )
                        .on_hover_text("Animation Duration");
                    });
                    ui.end_row();
                }

                ui.label("Background");
                let mut bg = [
                    state.render_state.settings.background_color.r as f32,
                    state.render_state.settings.background_color.g as f32,
                    state.render_state.settings.background_color.b as f32,
                    state.render_state.settings.background_color.a as f32,
                ];
                ui.color_edit_button_rgba_premultiplied(&mut bg);
                state.render_state.settings.background_color = wgpu::Color {
                    r: bg[0] as f64,
                    g: bg[1] as f64,
                    b: bg[2] as f64,
                    a: bg[3] as f64,
                };
                ui.end_row();

                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                    ui.add(egui::Label::new("Rendering Modes").wrap_mode(TextWrapMode::Extend));
                });
                ui.vertical(|ui| {
                    ui.checkbox(&mut state.render_state.settings.dvr.enabled, "Volume");
                    ui.checkbox(&mut state.render_state.settings.iso_surface.enabled, "Iso Surface");
                });

                ui.end_row();
                ui.label("Near Clip Plane");
                let volume_aabb = state.volume.volume.aabb;

                // find the distance from the camera to the aabb center in camera z direction
                let h = volume_aabb.center() - state.camera.position;
                let c = state.camera.view_direction();
                let a = h.angle(c).cos()*h.magnitude();

                let min_near = a-volume_aabb.radius();
                let max_far = a+volume_aabb.radius();
                let max_near = state.camera.projection.zfar;
                let min_far = state.camera.projection.znear;
                
                ui.add(egui::Slider::new(&mut state.camera.projection.znear, min_near..=max_near).clamp_to_range(true).fixed_decimals(3));
                ui.end_row();

                // ui.label("Far Clip Plane");
                // ui.add(egui::Slider::new(&mut state.camera.projection.zfar, min_far..=max_far).clamp_to_range(true));

                // workaround until clamp_to_range works with sliders agai
                state.camera.projection.znear = state.camera.projection.znear.clamp(min_near,max_near );
                state.camera.projection.zfar = state.camera.projection.zfar.clamp(min_far,max_far);

                ui.end_row();
            });

        CollapsingHeader::new("Presets").show_unindented(ui, |ui| {
            egui::Grid::new("presets grid").max_col_width(100.).num_columns(2).show(ui,|ui|{
            if state.presets.len() > 0{
                    ui.label("Load Preset");
                    egui::ComboBox::new("preset select", "")
                        .selected_text(
                            "<Select Preset>",)
                        .show_ui(ui, |ui| {
                            for (name,p) in &state.presets{
                                if ui.selectable_label(
                                    false,
                                    p.name.clone(),
                                ).clicked(){
                                    state.render_state.settings = p.render_settings.clone();
                                    state.selected_preset = Some(name.clone());
                                    if let Some(cmap) = &p.cmap{
                                        state.cmap = cmap.clone();
                                    }
                                    log::info!("Loaded preset {}", p.name);
                                }
                            }
                        });
            }else{
                ui.label("No presets available");
            }
            ui.end_row();

            ui.label("New Preset");

            let mut file_name = ctx.data_mut(|d| {
                let file_name: &mut String = d.get_temp_mut_or(Id::new("preset_name"), "my_preset".to_string());
                return file_name.clone();
            });
            TextEdit::singleline(&mut file_name ).show(ui);
            ctx.data_mut(|d| {
                d.insert_temp(Id::new("preset_name"), file_name.clone());
            });
            ui.end_row();
            ui.label("");
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Save to File").clicked(){
                let file = rfd::FileDialog::new().set_directory("./").add_filter("JSON File", &["json"]).save_file();
                if let Some(file) = file {
                    let file = std::fs::File::create(file).unwrap();
                    let preset = Preset{
                        name: file_name.clone(),
                        render_settings: state.render_state.settings.clone(),
                        cmap: state.render_state.settings.dvr.enabled.then_some(state.cmap.clone()),
                    };
                    serde_json::to_writer_pretty(file, &preset).unwrap();
                    log::info!("Saved present to {}", file_name);
                }
            }
            #[cfg(target_arch = "wasm32")]
            if ui.button("Save").clicked(){
                match crate::presets::Presets::from_local_storage(){
                    Ok(mut local_presets) => {
                        let new_preset = Preset{
                            name: file_name.clone(),
                            render_settings: state.render_state.settings.clone(),
                            cmap: state.render_state.settings.dvr.enabled.then_some(state.cmap.clone()),
                        };
                        local_presets.0.insert(file_name.clone(), new_preset.clone());
                        local_storage().unwrap().set_item("presets", &serde_json::to_string(&local_presets).unwrap()).unwrap();
                        state.presets.insert(file_name.clone(), new_preset);
                        log::info!("Saved present to local storage");
                    },
                    Err(err) => log::error!("failed to load presets from local storage: {}", err)
                }
                
            }
        });
        });

        CollapsingHeader::new("Advanced").show_unindented(ui, |ui| {
            Grid::new("settings_advanced")
                .num_columns(2)
                .show(ui, |ui| {
                    if state.render_state.settings.spatial_filter == wgpu::FilterMode::Linear{
                        ui.label("Step Size").on_hover_text("Distance between sample steps for rendering.\nSmaller values give better quality but are slower.");
                        ui.add(
                            egui::DragValue::new(&mut state.render_state.step_size)
                                .speed(0.01)
                                .range((1e-3)..=(0.1)),
                        );
                        ui.end_row();
                    }
                    ui.with_layout(Layout::top_down(Align::Min), |ui| {
                        ui.add(
                            egui::Label::new("Interpolation")
                                .selectable(true)
                                .wrap_mode(TextWrapMode::Extend),
                        )
                    });
                    ui.vertical(|ui| {
                        if state.volume.volume.timesteps() > 1 {
                            egui::ComboBox::new("temporal_interpolation", "Time")
                                .selected_text(match state.render_state.settings.temporal_filter {
                                    wgpu::FilterMode::Nearest => "Nearest",
                                    wgpu::FilterMode::Linear => "Linear",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut state.render_state.settings.temporal_filter,
                                        wgpu::FilterMode::Nearest,
                                        "Nearest",
                                    );
                                    ui.selectable_value(
                                        &mut state.render_state.settings.temporal_filter,
                                        wgpu::FilterMode::Linear,
                                        "Linear",
                                    )
                                });

                            ui.end_row();
                        }

                        egui::ComboBox::new("spatial_interpolation", "Space")
                            .selected_text(match state.render_state.settings.spatial_filter {
                                wgpu::FilterMode::Nearest => "Nearest",
                                wgpu::FilterMode::Linear => "Linear",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut state.render_state.settings.spatial_filter,
                                    wgpu::FilterMode::Nearest,
                                    "Nearest",
                                );
                                ui.selectable_value(
                                    &mut state.render_state.settings.spatial_filter,
                                    wgpu::FilterMode::Linear,
                                    "Linear",
                                )
                            });
                    });

                    ui.end_row();
                    ui.label("Bounding Box");
                    ui.checkbox(&mut state.show_box, "");
                });
        });
    });

    // let mut cmap = state.cmap.clone();

    if state.colormap_editor_visible && state.render_state.settings.dvr.enabled {
        egui::Window::new("Volume Rendering")
            .default_size(vec2(300., 50.))
            .show(ctx, |ui| {
                Grid::new("settings_volume").num_columns(2).show(ui,|ui|{
                    ui.label("Density Scale");
                    ui.add(
                        egui::DragValue::new(&mut state.render_state.settings.dvr.distance_scale)
                            .speed(0.01)
                            .range((1e-4)..=(100000.)),
                    );
                    ui.end_row();
                    ui.with_layout(Layout::top_down(Align::Min), |ui| {
                        ui.add(
                            egui::Label::new("Value Range")
                                .selectable(true)
                                .wrap_mode(TextWrapMode::Extend),
                        )
                    });
                    ui.vertical(|ui| {
                        let min_b = state
                        .render_state.settings.dvr
                        .vmin
                        .unwrap_or(state.volume.volume.min_value);
                        let max_b = state
                            .render_state.settings.dvr
                            .vmax
                            .unwrap_or(state.volume.volume.max_value);

                        let vmin_min = state.volume.volume.min_value.min(min_b);
                        let vmax_max = state.volume.volume.max_value.max(max_b);
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new("Min"));
                            optional_drag(
                                ui,
                                &mut state.render_state.settings.dvr.vmin,
                                Some(vmin_min..=max_b),
                                Some(0.01),
                                Some(vmin_min),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Max");
                            optional_drag(
                                ui,
                                &mut state.render_state.settings.dvr.vmax,
                                Some(min_b..=vmax_max),
                                Some(0.01),
                                Some(vmax_max),
                            );
                        });
                    });
                    ui.end_row();
                    #[cfg(feature = "colormaps")]
                    if state.cmap_select_visible {
                        ui.label("Colormap");
                        ui.horizontal(|ui| {
                            let cmaps = &COLORMAPS;
                            let mut selected_cmap: (String, String) = ui.ctx().data_mut(|d| {
                                d.get_persisted_mut_or(
                                    "selected_cmap".into(),
                                    ("seaborn".to_string(), "icefire".to_string()),
                                )
                                .clone()
                            });
                            let mut search_term: String = ui.ctx().data_mut(|d| {
                                d.get_temp_mut_or("cmap_search".into(), "".to_string())
                                    .clone()
                            });
                            let old_selected_cmap = selected_cmap.clone();
                            egui::ComboBox::new("cmap_select", "")
                                .selected_text(selected_cmap.1.clone())
                                .show_ui(ui, |ui| {
                                    ui.add(
                                        egui::text_edit::TextEdit::singleline(&mut search_term)
                                            .hint_text("Search..."),
                                    );
                                    for (group, cmaps) in cmaps.iter() {
                                        ui.label(group);
                                        let mut sorted_cmaps: Vec<_> = cmaps.iter().collect();
                                        sorted_cmaps.sort_by_key(|e| e.0);
                                        for (name, cmap) in sorted_cmaps {
                                            if name.contains(&search_term) {
                                                let texture =
                                                    load_or_create(ui, cmap, COLORMAP_RESOLUTION);
                                                ui.horizontal(|ui| {
                                                    ui.image(egui::ImageSource::Texture(
                                                        egui::load::SizedTexture {
                                                            id: texture,
                                                            size: vec2(50., 10.),
                                                        },
                                                    ));
                                                    ui.selectable_value(
                                                        &mut selected_cmap,
                                                        (group.clone(), name.clone()),
                                                        name,
                                                    );
                                                });
                                            }
                                        }
                                        ui.separator();
                                    }
                                });
                            if old_selected_cmap != selected_cmap {
                                let old_alpha = state.cmap.a.clone();
                                state.cmap = cmaps[&selected_cmap.0][&selected_cmap.1]
                                    .into_linear_segmented(COLORMAP_RESOLUTION);
                                if state.cmap.a.is_none()
                                    || cmaps[&selected_cmap.0][&selected_cmap.1]
                                        .has_boring_alpha_channel()
                                {
                                    state.cmap.a = old_alpha;
                                }
                                ui.ctx().data_mut(|d| {
                                    d.insert_persisted("selected_cmap".into(), selected_cmap);
                                });
                            }
                            ui.ctx()
                                .data_mut(|d| d.insert_temp("cmap_search".into(), search_term));
                            if state.cmap.a.is_none() {
                                state.cmap.a = Some(vec![(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
                            }
                            if ui.button("↔").clicked() {
                                state.cmap = (&state.cmap).reverse();
                            }
                        });
                    }
                });
                let vmin = state
                    .render_state.settings.dvr
                    .vmin
                    .unwrap_or(state.volume.volume.min_value);
                let vmax = state
                    .render_state.settings.dvr
                    .vmax
                    .unwrap_or(state.volume.volume.max_value);
                ui.add(egui::Label::new(egui::RichText::new("Preview").strong()));
                ui.end_row();
                show_cmap(ui, egui::Id::new("cmap preview"), &state.cmap, vmin, vmax);
                CollapsingHeader::new("Alpha Editor").default_open(true).show_unindented(ui, |ui| {
                    ui.end_row();
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Presets:");
                        let v_hack = ui
                            .button("\\/")
                            .on_hover_text("double click for smooth version");
                        if v_hack.clicked() {
                            state.cmap.a = Some(vec![(0.0, 1.0, 1.0), (0.5, 0., 0.), (1.0, 1.0, 1.0)]);
                        }
                        if v_hack.double_clicked() {
                            state.cmap.a =
                                Some(build_segments(25, |x| ((x * 2. * PI).cos() + 1.) / 2.));
                        }
                        let slope_hack = ui
                            .button("/")
                            .on_hover_text("double click for smooth version");
                        if slope_hack.clicked() {
                            state.cmap.a = Some(build_segments(2, |x| (-(x * PI).cos() + 1.) / 2.));
                        }
                        if slope_hack.double_clicked() {
                            state.cmap.a = Some(build_segments(25, |x| (-(x * PI).cos() + 1.) / 2.));
                        }
                        let double_v_hack = ui
                            .button("/\\/\\")
                            .on_hover_text("double click for smooth version");
                        if double_v_hack.clicked() {
                            state.cmap.a =
                                Some(build_segments(5, |x| (-(x * 4. * PI).cos() + 1.) / 2.));
                        }
                        if double_v_hack.double_clicked() {
                            state.cmap.a =
                                Some(build_segments(25, |x| (-(x * 4. * PI).cos() + 1.) / 2.));
                        }
                        if ui.button("-").clicked() {
                            state.cmap.a = Some(vec![(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
                        }
                    });

                    ui.separator();

                    if let Some(a) = &mut state.cmap.a {
                        tf_ui(ui, a)
                        .on_hover_text("Drag anchor points to change transfer function.\nLeft-Click for new anchor point.\nRight-Click to delete anchor point.");
                    }
                    ui.end_row();
                });

                ui.separator();
            });
    }
    state
        .cmap_gpu
        .update(&state.wgpu_context.queue, &state.cmap);

    if state.render_state.settings.iso_surface.enabled {
        egui::Window::new("Iso Surface Rendering")
            .default_size(vec2(300., 50.))
            .show(ctx, |ui: &mut Ui| {
                egui::Grid::new("iso_surface_settings")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Threshold");
                        ui.add(
                            egui::DragValue::new(
                                &mut state.render_state.settings.iso_surface.threshold,
                            )
                            .range(
                                state.volume.volume.min_value..=state.volume.volume.max_value, // TODO use all volume for vmin vmax
                            )
                            .speed(0.1),
                        );
                        ui.end_row();

                        ui.label("Color");
                        color_edit_button_rgba(
                            ui,
                            &mut state.render_state.settings.iso_surface.diffuse_color,
                        );
                        ui.end_row();

                        ui.label("SSAO");
                        ui.checkbox(&mut state.render_state.settings.ssao.enabled, "");
                        ui.end_row();
                    });

                ui.collapsing("Advanced", |ui| {
                    egui::collapsing_header::CollapsingHeader::new("Phong Shading")
                        .open(Some(true))
                        .show(ui, |ui| {
                            egui::Grid::new("iso_surface_settings_phong")
                                .num_columns(2)
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("Shininess");
                                    ui.add(
                                        egui::DragValue::new(
                                            &mut state.render_state.settings.iso_surface.shininess,
                                        )
                                        .range((0.)..=1e6),
                                    );
                                    ui.end_row();

                                    ui.label("Ambient Color");
                                    color_edit_button_rgb(
                                        ui,
                                        &mut state.render_state.settings.iso_surface.ambient_color,
                                    );
                                    ui.end_row();

                                    ui.label("Specular Color");
                                    color_edit_button_rgb(
                                        ui,
                                        &mut state.render_state.settings.iso_surface.specular_color,
                                    );
                                    ui.end_row();

                                    ui.label("Light Color");
                                    color_edit_button_rgb(
                                        ui,
                                        &mut state.render_state.settings.iso_surface.light_color,
                                    );
                                    ui.end_row();
                                });
                        });
                    egui::collapsing_header::CollapsingHeader::new("SSAO")
                        .open(Some(true))
                        .show(ui, |ui| {
                            egui::Grid::new("iso_surface_settings_ssao")
                                .num_columns(2)
                                .striped(true)
                                .show(ui, |ui| {
                                    if state.render_state.settings.ssao.enabled {
                                        ui.label("Radius");
                                        ui.add(egui::Slider::new(
                                            &mut state.render_state.settings.ssao.radius,
                                            0.01..=2.0,
                                        ));
                                        ui.end_row();
                                        ui.label("Bias");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut state.render_state.settings.ssao.bias,
                                                0.001..=0.2,
                                            )
                                            .logarithmic(true),
                                        );
                                        ui.end_row();
                                        ui.label("Kernel Size");
                                        ui.add(
                                            egui::Slider::new(
                                                &mut state.render_state.settings.ssao.kernel_size,
                                                1..=256,
                                            )
                                            .logarithmic(true),
                                        );
                                        ui.end_row();
                                    }
                                    if state.render_state.settings.spatial_filter
                                        == wgpu::FilterMode::Nearest
                                    {
                                        ui.label("Cube Surface Normal");
                                        ui.checkbox(
                                            &mut state
                                                .render_state
                                                .settings
                                                .iso_surface
                                                .use_cube_surface_grad,
                                            "",
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                });
            });
    }
    if state.volume_info_visible {
        egui::Window::new("Volume Info").show(ctx, |ui| {
            egui::Grid::new("volume_info")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Timesteps");
                    ui.label(state.volume.volume.timesteps().to_string());
                    ui.end_row();
                    ui.label("Resolution");
                    let res = state.volume.volume.resolution();
                    ui.label(format!("{}x{}x{} (WxHxD)", res.x, res.y, res.z));
                    ui.end_row();
                    ui.label("Value range");
                    ui.label(format!(
                        "[{} , {}]",
                        state.volume.volume.min_value, state.volume.volume.max_value
                    ));
                    ui.end_row();
                });
        });
    }

    let frame_rect = ctx.available_rect();
    egui::Area::new(egui::Id::new("bbox"))
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(0., 0.))
        .interactable(false)
        .order(Order::Background)
        .show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                vec2(frame_rect.width(), frame_rect.height()),
                Sense {
                    click: false,
                    drag: false,
                    focusable: false,
                },
            );

            let to_screen = emath::RectTransform::from_to(
                Rect::from_two_pos(Pos2::new(-1.0, -1.0), Pos2::new(1.0, 1.0)),
                response.rect,
            );
            let t = state.camera.proj_matrix() * state.camera.view_matrix();

            if state.show_box {
                // bbox
                let aabb = state.volume.volume.aabb;
                aabb.corners();
                let corners = aabb.corners().map(|p| {
                    let p_screen = t.transform_point(p);
                    to_screen.transform_pos(pos2(p_screen.x, p_screen.y))
                });
                let lines = [
                    (0, 1, egui::Color32::BLUE),
                    (0, 2, egui::Color32::GREEN),
                    (0, 4, egui::Color32::RED),
                    (6, 2, egui::Color32::YELLOW),
                    (6, 4, egui::Color32::YELLOW),
                    (6, 7, egui::Color32::YELLOW),
                    (5, 7, egui::Color32::YELLOW),
                    (5, 4, egui::Color32::YELLOW),
                    (5, 1, egui::Color32::YELLOW),
                    (3, 1, egui::Color32::YELLOW),
                    (3, 2, egui::Color32::YELLOW),
                    (3, 7, egui::Color32::YELLOW),
                ];
                for (a, b, color) in lines {
                    painter.add(PathShape::line(
                        vec![corners[a], corners[b]],
                        Stroke::new(3., color),
                    ));
                }
            }
            if state.controller.right_mouse_pressed {
                let cam_center = t.transform_point(state.controller.center);
                let cam_center_ui = to_screen.transform_pos(pos2(cam_center.x, cam_center.y));
                painter.add(egui::Shape::circle_stroke(
                    cam_center_ui,
                    10.,
                    Stroke::new(
                        3.,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100),
                    ),
                ));
                painter.add(egui::Shape::circle_stroke(
                    cam_center_ui,
                    13.,
                    Stroke::new(3., egui::Color32::from_rgba_unmultiplied(0, 0, 0, 100)),
                ));
            }
        });

    // check if the selected preset is still valid
    // if let Some(preset) = &state.selected_preset {
    //     let curr_preset = &state.presets[preset];
    //     let cmap_same = match &curr_preset.cmap {
    //         Some(cmap) => state.cmap == *cmap,
    //         None => state.cmap.a.is_none(),
    //     };
    //     if curr_preset.render_settings != state.render_state.settings || !cmap_same {
    //         state.selected_preset = None;
    //     }
    // }

    return ctx.has_requested_repaint();
}

use cgmath::{Angle, InnerSpace, Transform, Vector3, Vector4};
use egui::{epaint::PathShape, *};

pub fn tf_ui(ui: &mut Ui, points: &mut Vec<(f32, f32, f32)>) -> egui::Response {
    let (response, painter) = ui.allocate_painter(
        vec2(ui.available_width(), 100.),
        Sense::hover().union(Sense::click()),
    );

    let to_screen = emath::RectTransform::from_to(
        Rect::from_two_pos(Pos2::ZERO, Pos2::new(1., 1.)),
        response.rect,
    );

    let stroke = Stroke::new(1.0, Color32::from_rgb(25, 200, 100));
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let pp = to_screen.inverse().transform_pos(pos);
            let pp = (pp.x, 1. - pp.y, 1. - pp.y);
            let idx = points
                .iter()
                .enumerate()
                .find_map(|p| if p.1 .0 > pp.0 { Some(p.0) } else { None })
                .unwrap_or(points.len() - 1);
            points.insert(idx, pp);
        }
    }

    let control_point_radius = 8.0;

    let n = points.len();
    let mut new_points = Vec::with_capacity(n);
    let mut control_point_shapes = Vec::with_capacity(n);
    for (i, point) in points.iter().enumerate() {
        let size = Vec2::splat(2.0 * control_point_radius);
        let pos = pos2(point.0, 1. - point.1);
        let point_in_screen = to_screen.transform_pos(pos);
        let point_rect = Rect::from_center_size(point_in_screen, size);
        let point_id = response.id.with(i);
        let point_response = ui.interact(point_rect, point_id, Sense::drag().union(Sense::click()));

        let is_edge = i == 0 || i == n - 1;

        if !point_response.secondary_clicked() || is_edge {
            let e = 1e-3;
            let mut t = point_response.drag_delta();
            if is_edge {
                // cant move last and first point
                t.x = 0.;
            }
            // point cannot move past its neighbors
            let left = if i == 0 { 0. } else { points[i - 1].0 + e };
            let right = if i == n - 1 { 1. } else { points[i + 1].0 - e };
            let bbox = Rect::from_min_max(Pos2::new(left, 0.), Pos2::new(right, 1.));

            let mut new_point = pos2(point.0, 1. - point.1);
            new_point += to_screen.inverse().scale() * t;
            new_point = to_screen.from().intersect(bbox).clamp(new_point);
            new_points.push((new_point.x, 1. - new_point.y, 1. - new_point.y));

            let point_in_screen = to_screen.transform_pos(new_point);
            let stroke = ui.style().interact(&point_response).fg_stroke;

            control_point_shapes.push(Shape::circle_stroke(
                point_in_screen,
                control_point_radius,
                stroke,
            ));
        }
    }
    points.drain(0..n);
    points.extend(new_points);

    let points_in_screen: Vec<Pos2> = points
        .iter()
        .map(|p| to_screen.transform_pos(pos2(p.0, 1. - p.1)))
        .collect();

    painter.add(PathShape::line(points_in_screen, stroke));
    painter.extend(control_point_shapes);
    response
}

use std::hash::Hash;

/// Load or create a texture from a colormap
/// If the texture is already loaded, return the texture id
/// Otherwise, create a new texture and store it in the egui context
#[cfg(feature = "colormaps")]
fn load_or_create<C>(ui: &egui::Ui, cmap: C, n: u32) -> egui::TextureId
where
    C: ColorMap + Hash,
{
    let id = Id::new(&cmap);
    let tex: Option<egui::TextureHandle> = ui.ctx().data_mut(|d| d.get_temp(id));
    match tex {
        Some(tex) => tex.id(),
        None => {
            let tex = ui.ctx().load_texture(
                id.value().to_string(),
                egui::ColorImage::from_rgba_unmultiplied(
                    [n as usize, 1],
                    bytemuck::cast_slice(&cmap.rasterize(n as usize)),
                ),
                egui::TextureOptions::LINEAR,
            );
            let tex_id = tex.id();
            ui.ctx().data_mut(|d| d.insert_temp(id, tex));
            return tex_id;
        }
    }
}

// stores colormap texture in egui context
// only updates texture if it changed
fn cmap_preview<C>(ui: &egui::Ui, cmap: C, n: u32) -> egui::TextureId
where
    C: ColorMap + Hash,
{
    let id: Id = Id::new("cmap_preview_texture");
    let tex: Option<(Id, egui::TextureHandle)> = ui.ctx().data_mut(|d| d.get_temp(id));
    match tex {
        Some((old_id, mut tex)) => {
            if old_id != id.with(&cmap) {
                tex.set(
                    egui::ColorImage::from_rgba_unmultiplied(
                        [n as usize, 1],
                        bytemuck::cast_slice(&cmap.rasterize(n as usize)),
                    ),
                    egui::TextureOptions::LINEAR,
                );
            }
            tex.id()
        }
        None => {
            let tex = ui.ctx().load_texture(
                id.value().to_string(),
                egui::ColorImage::from_rgba_unmultiplied(
                    [n as usize, 1],
                    bytemuck::cast_slice(&cmap.rasterize(n as usize)),
                ),
                egui::TextureOptions::LINEAR,
            );
            let tex_id = tex.id();
            ui.ctx()
                .data_mut(|d| d.insert_temp(id, (id.with(cmap), tex)));
            return tex_id;
        }
    }
}

fn optional_drag<T: Numeric>(
    ui: &mut egui::Ui,
    opt: &mut Option<T>,
    range: Option<RangeInclusive<T>>,
    speed: Option<impl Into<f64>>,
    default: Option<T>,
) {
    let mut placeholder = default.unwrap_or(T::from_f64(0.));
    let mut drag = if let Some(ref mut val) = opt {
        egui_winit::egui::DragValue::new(val)
    } else {
        egui_winit::egui::DragValue::new(&mut placeholder).custom_formatter(|_, _| {
            if let Some(v) = default {
                format!("{:.2}", v.to_f64())
            } else {
                "—".into()
            }
        })
    };
    if let Some(range) = range {
        drag = drag.range(range);
    }
    if let Some(speed) = speed {
        drag = drag.speed(speed);
    }
    let changed = ui.add(drag).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(placeholder);
    }
}

fn show_cmap(ui: &mut egui::Ui, id: egui::Id, cmap: impl ColorMap + Hash, vmin: f32, vmax: f32) {
    let texture = cmap_preview(ui, cmap, COLORMAP_RESOLUTION);
    let width = vmax - vmin;
    let height = width / 5.;
    let image = PlotImage::new(
        texture,
        PlotPoint::new(vmin + width * 0.5, height / 2.),
        vec2(width, height),
    );
    let plot = Plot::new(id)
        .show_x(true)
        .show_y(false)
        .height(100.)
        .show_background(false)
        .show_grid(false)
        .set_margin_fraction(Vec2::zeroed())
        .custom_y_axes(vec![])
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_zoom(false);
    plot.show(ui, |plot_ui| {
        plot_ui.image(image);
    });
}

fn build_segments<F: Fn(f32) -> f32>(n: usize, f: F) -> Vec<(f32, f32, f32)> {
    (0..n)
        .map(|i| {
            let x = i as f32 / (n as f32 - 1.);
            let v = f(x);
            (x, v, v)
        })
        .collect()
}

pub fn color_edit_button_rgb(ui: &mut Ui, rgb: &mut Vector3<f32>) -> Response {
    let mut rgba = Rgba::from_rgb(rgb[0], rgb[1], rgb[2]);
    let response = color_picker::color_edit_button_rgba(ui, &mut rgba, color_picker::Alpha::Opaque);
    rgb[0] = rgba[0];
    rgb[1] = rgba[1];
    rgb[2] = rgba[2];
    response
}

pub fn color_edit_button_rgba(ui: &mut Ui, rgb: &mut Vector4<f32>) -> Response {
    let mut rgba = Rgba::from_rgba_premultiplied(rgb[0], rgb[1], rgb[2], rgb[3]);
    let response =
        color_picker::color_edit_button_rgba(ui, &mut rgba, color_picker::Alpha::BlendOrAdditive);
    rgb[0] = rgba[0];
    rgb[1] = rgba[1];
    rgb[2] = rgba[2];
    rgb[3] = rgba[3];
    response
}
