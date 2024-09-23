use ::ndarray::{Array, IxDyn};
use bytemuck::Zeroable;
use cgmath::{BaseNum, EuclideanSpace, MetricSpace, Point3, Vector3};
use half::f16;
#[cfg(target_arch = "wasm32")]
use instant::Instant;
use ndarray::{Array4, ArrayBase, Axis, Data, Ix4, OwnedRepr};
use nifti::{InMemNiftiObject, IntoNdArray, NiftiObject};
use npyz::{npz, Deserialize, NpyFile};
use num_traits::Float;
use std::io::{Read, Seek};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use wgpu::util::{DeviceExt, TextureDataOrder};

pub struct Volume {
    // pub timesteps: u32,
    // pub resolution: Vector3<u32>,
    pub aabb: Aabb<f32>,
    pub min_value: f32,
    pub max_value: f32,
    data: ndarray::Array4<f16>,
}

impl Volume {
    pub fn from_array(mut data: ArrayBase<OwnedRepr<f16>, IxDyn>) -> Result<Volume, anyhow::Error> {
        let dim = data.shape().len();
        if dim != 3 && dim != 4 {
            // if we can squeeze the array, continue
            // otherwise, bail
            data = squeeze(data);
            let new_dim = data.shape().len();
            if new_dim != 3 && new_dim != 4 {
                anyhow::bail!("unsupported shape: {:?}", data.shape());
            }
        }

        // add a time axis if not present
        if data.shape().len() == 3 {
            data = data.insert_axis(ndarray::Axis(0));
        }

        let data = data.as_standard_layout();

        let shape = data.shape();

        let res_min = shape.iter().skip(1).min().unwrap();

        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(
                shape[3] as f32 / *res_min as f32,
                shape[2] as f32 / *res_min as f32,
                shape[1] as f32 / *res_min as f32,
            ),
        };
        log::info!("volume shape is: {:?}, interpreted as [TxWxHxD]", shape);

        let mut min_value = f32::MAX;
        let mut max_value = f32::MIN;

        let array_f16 = data.map(|v| {
            let v_f: f64 = (*v).clone().into();
            min_value = min_value.min(v_f as f32);
            max_value = max_value.max(v_f as f32);
            f16::from_f64(v_f)
        });

        if min_value == max_value {
            log::warn!(
                "min value({}) == max value({}), setting max to min + 1",
                min_value,
                max_value
            );
            max_value = min_value + 1.0;
        }

        log::debug!("a shape: {:?}", shape);
        let volume = Self {
            aabb,
            min_value: min_value as f32,
            max_value: max_value as f32,
            data: Array4::from_shape_vec(
                Ix4(shape[0], shape[1], shape[2], shape[3]),
                array_f16.as_slice().unwrap().to_vec(),
            )
            .unwrap(),
        };

        return Ok(volume);
    }

    pub fn timesteps(&self) -> usize {
        self.data.shape()[0]
    }

    pub fn resolution(&self) -> Vector3<u32> {
        Vector3::new(
            self.data.shape()[1] as u32,
            self.data.shape()[2] as u32,
            self.data.shape()[3] as u32,
        )
    }

    pub fn load<'a, R>(mut reader: R) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let start = Instant::now();
        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer)?;
        let is_npz = buffer == *b"\x50\x4B\x03\x04";

        reader.seek(std::io::SeekFrom::Start(344))?;
        reader.read_exact(&mut buffer)?;
        let is_nifti = buffer == *b"\x6E\x2B\x31\x00";
        reader.seek(std::io::SeekFrom::Start(0))?;

        let data = if is_nifti {
            Self::load_nifti(reader)
        } else if is_npz {
            Self::load_npz(reader)
        } else {
            Self::load_npy(reader)
        }?;
        let volume = Self::from_array(data);
        log::info!("loading volume took: {:?}", start.elapsed());
        return volume;
    }

    fn load_npz<'a, R>(reader: R) -> anyhow::Result<ArrayBase<OwnedRepr<f16>, IxDyn>>
    where
        R: Read + Seek,
    {
        let mut reader = npz::NpzArchive::new(reader)?;
        let arr_name = reader
            .array_names()
            .next()
            .ok_or(anyhow::format_err!("no array present"))?
            .to_string();
        let array = reader.by_name(arr_name.as_str())?.unwrap();
        Self::read_npy(array)
    }

    fn load_npy<'a, R>(reader: R) -> anyhow::Result<ArrayBase<OwnedRepr<f16>, IxDyn>>
    where
        R: Read + Seek,
    {
        let array = NpyFile::new(reader)?;
        Self::read_npy(array)
    }

    fn read_npy_dyn<'a, R, P>(
        npy_file: NpyFile<R>,
    ) -> anyhow::Result<ArrayBase<OwnedRepr<f16>, IxDyn>>
    where
        R: Read,
        P: Into<f64> + Deserialize + Clone,
    {
        let shape: Vec<usize> = npy_file.shape().iter().map(|v| *v as usize).collect();
        let data = npy_file.into_vec::<P>()?;
        let array_s = Array::from_shape_vec(IxDyn(&shape), data)?;
        return Ok(array_s.map(|v| f16::from_f64((*v).clone().into())));
    }

    fn read_npy<'a, R>(array: NpyFile<R>) -> anyhow::Result<ArrayBase<OwnedRepr<f16>, IxDyn>>
    where
        R: Read,
    {
        match array.dtype() {
            npyz::DType::Plain(d) => match d.type_char() {
                npyz::TypeChar::Float => match d.num_bytes().unwrap() {
                    2 => Self::read_npy_dyn::<_, f16>(array),
                    4 => Self::read_npy_dyn::<_, f32>(array),
                    8 => Self::read_npy_dyn::<_, f64>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                npyz::TypeChar::Uint => match d.num_bytes().unwrap() {
                    1 => Self::read_npy_dyn::<_, u8>(array),
                    2 => Self::read_npy_dyn::<_, u16>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                npyz::TypeChar::Int => match d.num_bytes().unwrap() {
                    1 => Self::read_npy_dyn::<_, i8>(array),
                    2 => Self::read_npy_dyn::<_, i16>(array),
                    _ => anyhow::bail!("unsupported type {:}", d),
                },
                _ => anyhow::bail!("unsupported type {:}", d),
            },
            d => anyhow::bail!("unsupported type {:}", d.descr()),
        }
    }

    fn load_nifti<'a, R>(reader: R) -> anyhow::Result<ArrayBase<OwnedRepr<f16>, IxDyn>>
    where
        R: Read + Seek,
    {
        let obj = InMemNiftiObject::from_reader(reader)?;
        let volume = obj.into_volume();

        let data = volume.into_ndarray::<f32>()?;
        return Ok(data.map(|v| f16::from_f64((*v).clone().into())));
    }
}

pub struct VolumeGPU {
    pub(crate) textures: Vec<wgpu::Texture>,
    pub(crate) volume: Volume,
}

impl VolumeGPU {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, volume: Volume) -> Self {
        let timesteps = volume.timesteps();
        let resolution = volume.resolution();
        let textures = (0..timesteps)
            .map(|i| {
                device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some(format!("volume texture {}", i).as_str()),
                        size: wgpu::Extent3d {
                            width: resolution[2],
                            height: resolution[1],
                            depth_or_array_layers: resolution[0],
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::R16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(
                        &volume
                            .data
                            .index_axis(Axis(0), i as usize)
                            .as_slice()
                            .unwrap(),
                    ),
                )
            })
            .collect();
        Self { textures, volume }
    }
}

#[repr(C)]
#[derive(Zeroable, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Aabb<F: Float + BaseNum + serde::Serialize> {
    pub min: Point3<F>,
    pub max: Point3<F>,
}

impl<F: Float + BaseNum + serde::Serialize> Aabb<F> {
    pub fn unit() -> Self {
        Self {
            min: Point3::new(F::zero(), F::zero(), F::zero()),
            max: Point3::new(F::one(), F::one(), F::one()),
        }
    }

    pub fn center(&self) -> Point3<F> {
        self.min.midpoint(self.max)
    }

    /// radius of a sphere that contains the aabb
    pub fn radius(&self) -> F {
        self.min.distance(self.max) / (F::one() + F::one())
    }

    /// returns box corners in order:
    /// min, max_z, max_y, max_yz, max_x, max_xz, max_xy, max
    pub fn corners(&self) -> [Point3<F>; 8] {
        [
            self.min,
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            self.max,
        ]
    }
}

/// Squeeze out all dimensions of size 1
pub fn squeeze<A, S>(array: ArrayBase<S, IxDyn>) -> ArrayBase<S, IxDyn>
where
    S: Data<Elem = A>,
{
    let mut out = array;
    for axis in (0..out.shape().len()).rev() {
        if out.shape()[axis] == 1 && out.shape().len() > 1 {
            out = out.remove_axis(Axis(axis));
        }
    }
    out
}
