
struct Aabb {
    @align(16) min: vec3<f32>,
    @align(16) max: vec3<f32>,
}

struct Settings {
    volume_aabb: Aabb,

    time: f32,
    time_steps: u32,
    temporal_filter: u32,
    spatial_filter: u32,
    
    distance_scale: f32,
    vmin: f32,
    vmax: f32,
    gamma_correction: u32,
    
    @align(16) @size(16) iso_ambient_color: vec3<f32>,
    @align(16) @size(16) iso_specular_color: vec3<f32>,
    @align(16) @size(16) iso_light_color: vec3<f32>,
    iso_diffuse_color: vec4<f32>,

    render_mode_volume: u32, // use volume rendering
    render_mode_iso: u32, // use iso rendering
    use_cube_surface_grad: u32, // whether to use cube surface gradients for render_mode_iso_nearest
    iso_shininess: f32,

    ssao_enabled:u32,
    ssao_radius: f32,
    ssao_bias: f32,
    ssao_kernel_size:u32,
    
    background_color:vec4<f32>,

    iso_threshold: f32,
    step_size:f32,
}



@group(0) @binding(4)
var<uniform> settings: Settings;


@group(1) @binding(0)
var texture_dvr : texture_2d<f32>;
@group(1) @binding(1)
var texture_iso : texture_2d<f32>;
@group(1) @binding(2)
var texture_ssao : texture_2d<f32>;
@group(1) @binding(3)
var texture_sampler: sampler;


struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOut {

    // creates two vertices that cover the whole screen
    let xy = vec2<f32>(
        f32(in_vertex_index % 2u == 0u),
        f32(in_vertex_index < 2u)
    );
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), vec2<f32>(xy.x, 1. - xy.y));
}

@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let color_dvr = textureSample(texture_dvr, texture_sampler, vertex_in.tex_coord);
    var color_iso = textureSample(texture_iso, texture_sampler, vertex_in.tex_coord);
    let color_ssao = textureSample(texture_ssao, texture_sampler, vertex_in.tex_coord).r;

    if bool(settings.ssao_enabled) && color_ssao > 0. {
        color_iso = vec4<f32>(color_iso.rgb*(1.-color_ssao),color_iso.a);
    }

    var color = blend(color_dvr,color_iso);
    return blend(color, settings.background_color);
}

// blends two premultiplied colors
fn blend(src: vec4<f32>, dst: vec4<f32>) -> vec4<f32> {
    return src + dst * (1. - src.a);
}