const FILTER_NEAREST:u32 = 0;
const FILTER_LINEAR:u32 = 1;

const PI:f32 = 3.1415926535897932384626433832795;
const TWO_PI:f32 = 6.283185307179586476925286766559;


const ISO_IRRADI_PERP: f32 = 1.0;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

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


struct Aabb {
    @align(16) min: vec3<f32>,
    @align(16) max: vec3<f32>,
}

struct Ray {
    orig: vec3<f32>,
    dir: vec3<f32>
};

/// adapted from https://github.com/evanw/webgl-path-tracing/blob/master/webgl-path-tracing.js
fn intersectAABB(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let tMin = (box_min - ray.orig) / ray.dir;
    let tMax = (box_max - ray.orig) / ray.dir;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);
    return vec2<f32>(t_near, t_far);
}

// ray is created based on view and proj matrix so
// that it matches the rasterizer used for drawing other stuff
fn create_ray(view_inv: mat4x4<f32>, proj_inv: mat4x4<f32>, px: vec2<f32>) -> Ray {
    var far = vec4<f32>((px * 2. - (1.)), 1., 1.);
    // depth prepass location
    var far_w = view_inv * proj_inv * far;
    far_w /= far_w.w + 1e-6;


    var near = vec4<f32>((px * 2. - (1.)), -1., 1.);
    // depth prepass location
    var near_w = view_inv * proj_inv * near;
    near_w /= near_w.w + 1e-6;

    return Ray(
        near_w.xyz,
        far_w.xyz - near_w.xyz,
    );
}

@group(0) @binding(0)
var volume : texture_3d<f32>;
@group(0) @binding(1)
var volume_next : texture_3d<f32>;
@group(0) @binding(2)
var volume_sampler: sampler;

@group(0) @binding(3)
var<uniform> camera: CameraUniforms;

@group(0) @binding(4)
var<uniform> settings: Settings;

@group(1) @binding(0)
var cmap : texture_2d<f32>;
@group(1) @binding(1)
var cmap_sampler: sampler;

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

// performs a step and returns the NDC cooidinate for the volume sampling
fn next_pos(pos: ptr<function,vec3<f32>>, step_size: f32, ray_dir: vec3<f32>) -> vec3<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    let sample_pos = ((*pos) - aabb.min) / aabb_size;
    *pos += ray_dir * step_size;
    return sample_pos;
}

fn sample_volume_gradient(pos: vec3<f32>) -> vec3<f32> {
    let size = vec3<f32>(textureDimensions(volume));
    let dx = vec3<f32>(1. / size.x, 0., 0.) * 2.;
    let dy = vec3<f32>(0., 1. / size.y, 0.) * 2.;
    let dz = vec3<f32>(0., 0., 1. / size.z) * 2.;
    let grad_x = (sample_volume(pos + dx / 2.) - sample_volume(pos - dx / 2.));
    let grad_y = (sample_volume(pos + dy / 2.) - sample_volume(pos - dy / 2.));
    let grad_z = (sample_volume(pos + dz / 2.) - sample_volume(pos - dz / 2.));
    return vec3<f32>(grad_x, grad_y, grad_z);
}

fn sample_volume(pos: vec3<f32>) -> f32 {
    //  origin is in bottom left corner so we need to flip y 
    let pos_m = vec3<f32>(pos.x, 1. - pos.y, pos.z);
    let sample_curr = textureSampleLevel(volume, volume_sampler, pos_m, 0.).r;
    let sample_next = textureSampleLevel(volume_next, volume_sampler, pos_m, 0.).r;
    if settings.temporal_filter == FILTER_NEAREST {
        return sample_curr;
    } else {
        let time_fraction = fract(settings.time * f32(settings.time_steps - (1)));
        return mix(sample_curr, sample_next, time_fraction);
    }
}

fn sample_cmap(value: f32) -> vec4<f32> {
    let value_n = (value - settings.vmin) / (settings.vmax - settings.vmin);
    return textureSampleLevel(cmap, cmap_sampler, vec2<f32>(value_n, 0.5), 0.);
}


fn phongBRDF(lightDir: vec3<f32>, viewDir: vec3<f32>, normal: vec3<f32>, phongDiffuseCol: vec3<f32>, phongSpecularCol: vec3<f32>, phongShininess: f32) -> vec3<f32> {
    var color = phongDiffuseCol;
    let reflectDir = reflect(-lightDir, normal);
    let specDot = max(dot(reflectDir, viewDir), 0.0);
    color += pow(specDot, phongShininess) * phongSpecularCol;
    return color;
}


struct DDAState {
    t_max: vec3<f32>,
    t_delta: vec3<f32>,
    step: vec3<i32>,
    voxel_index: vec3<i32>,
    step_dir_curr: vec3<i32>,
}

fn get_voxel_segment_length(ray_origin: vec3<f32>, ray_direction: vec3<f32>, voxel_index: vec3<i32>) -> f32 {
    let lower = vec3<f32>(voxel_index);
    let upper = vec3<f32>(voxel_index + 1);
    var t_near = -1e7;
    var t_far = 1e7;
    for (var i: i32 = 0; i < 3; i += 1) {
        if abs(ray_direction[i]) < 1e-4 {
            if ray_origin[i] < lower[i] || ray_origin[i] > upper[i] {
                return 0.0;
            }
        } else {
            var t0 = (lower[i] - ray_origin[i]) / ray_direction[i];
            var t1 = (upper[i] - ray_origin[i]) / ray_direction[i];
            if (t0 > t1) {
                let tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            if t0 > t_near {
                t_near = t0;
            }
            if t1 < t_far {
                t_far = t1;
            }
            if t_near > t_far || t_far < 0 {
                return 0.0;
            }
        }
    }
    return t_far - t_near;
}

fn next_pos_dda(
        pos: ptr<function,vec3<f32>>, step_size: ptr<function,f32>, ray_orig: vec3<f32>, ray_dir: vec3<f32>,
        state: ptr<function,DDAState>) -> vec3<f32> {
    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;

    let sample_pos = (vec3<f32>((*state).voxel_index) + 0.5) / vec3<f32>(textureDimensions(volume));

    *step_size = get_voxel_segment_length(ray_orig, ray_dir, (*state).voxel_index) / f32(textureDimensions(volume).x) * aabb_size.x;

    if ((*state).t_max.x < (*state).t_max.y) {
        if ((*state).t_max.x < (*state).t_max.z) {
            (*state).voxel_index.x += (*state).step.x;
            (*state).t_max.x += (*state).t_delta.x;
            (*state).step_dir_curr = vec3<i32>((*state).step.x, 0, 0);
          } else {
            (*state).voxel_index.z += (*state).step.z;
            (*state).t_max.z += (*state).t_delta.z;
            (*state).step_dir_curr = vec3<i32>(0, 0, (*state).step.z);
        }
    } else {
        if ((*state).t_max.y < (*state).t_max.z) {
            (*state).voxel_index.y += (*state).step.y;
            (*state).t_max.y += (*state).t_delta.y;
            (*state).step_dir_curr = vec3<i32>(0, (*state).step.y, 0);
        } else {
            (*state).voxel_index.z += (*state).step.z;
            (*state).t_max.z += (*state).t_delta.z;
            (*state).step_dir_curr = vec3<i32>(0, 0, (*state).step.z);
        }
    }

    *pos += ray_dir * (*step_size);

    return sample_pos;
}

fn intersect_aabb_dir(ray_origin: vec3<f32>, ray_direction: vec3<f32>, lower: vec3<f32>, upper: vec3<f32>) -> i32 {
    var dir = 0;
    var t_near_best = -1e7;
    for (var i: i32 = 0; i < 3; i += 1) {
        var t0 = (lower[i] - ray_origin[i]) / ray_direction[i];
        var t1 = (upper[i] - ray_origin[i]) / ray_direction[i];
        if (t0 > t1) {
            let tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        let t_near = t0;
        if t_near > t_near_best {
            t_near_best = t_near;
            dir = i;
        }
    }
    return dir;
}


// traces ray trough volume and returns color
fn trace_ray(ray_in: Ray,color_dvr:ptr<function,vec4<f32>>, color_iso:ptr<function,vec4<f32>>, normal_depth: ptr<function,vec4<f32>>){
    let ray_len = length(ray_in.dir);
    var ray = ray_in;
    ray.dir = normalize(ray.dir);

    let aabb = settings.volume_aabb;
    let aabb_size = aabb.max - aabb.min;
    
    let slice_min = vec3<f32>(0.);
    let slice_max = vec3<f32>(1.);
    // find closest point on volume
    let aabb_min = (aabb.min + (slice_min * aabb_size)); //  zxy for tensorf alignment
    let aabb_max = (aabb.max - ((1. - slice_max) * aabb_size)); //  zxy for tensorf alignment
    let intersec = intersectAABB(ray, aabb_min, aabb_max);

    if intersec.x > intersec.y {
        return;// return settings.background_color;
    }

    let start_cam_pos = ray.orig;
    let start = max(0., intersec.x) + 1e-4;
    ray.orig += start * ray.dir;

    var iters = 0u;
    var color = vec3<f32>(0.);
    var transmittance = 1.;

    let volume_size = textureDimensions(volume);

    var distance_scale = settings.distance_scale;

    var pos = ray.orig;

    let early_stopping_t = 1. / 255.;
    var step_size = settings.step_size;
    var sample_pos: vec3<f32> = next_pos(&pos, 0., ray.dir);
    var last_sample_pos = sample_pos;
    var last_sample = sample_volume(last_sample_pos);
    var normal = vec3<f32>(0.);
    var last_normal = vec3<f32>(0.);
    var depth = 1e6;

    let cam_pos = camera.view_inv[3].xyz;

    var state: DDAState;
    var start_point: vec3<f32>;
    if settings.spatial_filter == FILTER_NEAREST {
        start_point = (ray.orig - aabb.min) / aabb_size * vec3<f32>(volume_size);
        let end_point = (start_cam_pos + (intersec.y - 1e-4) * ray.dir - aabb.min) / aabb_size * vec3<f32>(volume_size);

        for (var i: i32 = 0; i < 3; i += 1) {
            state.step[i] = i32(sign(end_point[i] - start_point[i]));
            if state.step[i] != 0 {
                state.t_delta[i] = min(f32(state.step[i]) / (end_point[i] - start_point[i]), 1e7);
            } else {
                state.t_delta[i] = 1e7; // inf
            }
            if state.step[i] > 0 {
                state.t_max[i] = state.t_delta[i] * (1.0 - fract(start_point[i]));
            } else {
                state.t_max[i] = state.t_delta[i] * fract(start_point[i]);
            }
            state.voxel_index[i] = i32(floor(start_point[i]));
        }

        state.step_dir_curr = vec3<i32>(0, 0, 0);
        let dir_init = intersect_aabb_dir(start_point, ray.dir, aabb_min, aabb_max);
        state.step_dir_curr[dir_init] = state.step[dir_init];
    }

    // used for iso surface rendering
    var first = true;
    var sign = -1.;
    loop{
        if settings.spatial_filter == FILTER_NEAREST {
            sample_pos = next_pos_dda(&pos, &step_size, start_point, ray.dir, &state);
        } else {
            sample_pos = next_pos(&pos, step_size, ray.dir);
        }

        let slice_test = any(sample_pos < slice_min) || any(sample_pos > slice_max) ;
        if slice_test || distance(ray_in.orig,pos) > ray_len || iters > 10000 {
            break;
        }

        let sample = sample_volume(sample_pos);

        
        if bool(settings.render_mode_iso) {
            let iso_threshold = settings.iso_threshold;
            let new_sign = sign(sample - iso_threshold);

            if sign != new_sign {
                let t = (iso_threshold - last_sample) / (sample - last_sample + 1e-4);
                let intersection = mix(last_sample_pos, sample_pos.xyz, t);

                let view_dir = ray.dir;
                let light_dir = normalize(ray.dir + vec3<f32>(0.1));

                if bool(settings.use_cube_surface_grad) && settings.spatial_filter == FILTER_NEAREST {
                    normal = last_normal;
                } else {
                    let gradient = normalize(sample_volume_gradient(intersection));
                    if first {
                        normal = light_dir;
                    } else {
                        normal = -sign * gradient;
                    }
                }

                let diffuse_color = settings.iso_diffuse_color;
                let ambient_color = settings.iso_ambient_color;
                let specular_color = settings.iso_specular_color;
                let shininess = settings.iso_shininess;
                let light_color = settings.iso_light_color;
                let irradi_perp = ISO_IRRADI_PERP;

                var radiance = ambient_color;

                let irradiance = max(dot(light_dir, normal), 0.0) * irradi_perp;
                if !first{
                    if irradiance > 0.0 {
                        let brdf = phongBRDF(light_dir, view_dir, normal, diffuse_color.rgb, specular_color, shininess);
                        radiance += brdf * irradiance * light_color;
                    }
                }else{
                    radiance += diffuse_color.rgb;
                }

                depth = distance(start_cam_pos, pos);
                *color_iso = vec4<f32>(radiance,1.);

                break;
            }

            first = false;
            sign = new_sign;
        }
        if bool(settings.render_mode_volume) {
            let color_tf = sample_cmap(sample);
            // we dont want full opacity color
            let sigma = color_tf.a * (1. - 1e-6);
            if sigma > 0. {
                var sample_color = color_tf.rgb;
                let a_i = 1. - pow(1. - sigma, step_size * distance_scale);
                color += transmittance * a_i * sample_color;
                transmittance *= 1. - a_i;
            }
        }

        if transmittance <= early_stopping_t {
            break;
        }
        // check if within slice

        iters += 1u;
        last_sample = sample;
        last_sample_pos = sample_pos;
        last_normal = vec3<f32>(state.step_dir_curr);
    }
    if length(normal) < 1e-4 {
        normal = vec3<f32>(0.0, 0.0, -1.0);
        depth = 0.0;
    } else {
        normal = (camera.view * vec4<f32>(normal, 0.0)).xyz;
    }

    // write color and normal and depth values
    *normal_depth = vec4<f32>(normal, depth);
    *color_dvr = vec4<f32>(color, 1. - transmittance);
}

fn gamma_correction(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(pow(color.rgb, vec3<f32>(1. / 2.2)), color.a);
}


struct FragmentOut{
    @location(0) color_dvr:vec4<f32>,
    @location(1) color_iso:vec4<f32>,
    @location(2) normal_depth:vec4<f32>,
}

@fragment
fn fs_main(vertex_in: VertexOut) -> FragmentOut {
    let r_pos = vec2<f32>(vertex_in.tex_coord.x, vertex_in.tex_coord.y);
    let ray = create_ray(camera.view_inv, camera.proj_inv, r_pos);
    var normal_depth = vec4<f32>(0.);
    var color_iso = vec4<f32>(0.);
    var color_dvr = vec4<f32>(0.);
    trace_ray(ray,&color_dvr,&color_iso,&normal_depth);

    return FragmentOut(color_dvr,color_iso,normal_depth);
}