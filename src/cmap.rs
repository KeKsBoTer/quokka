use std::{
    hash::{Hash, Hasher},
    io::{Read, Seek, SeekFrom},
};

#[cfg(feature = "colormaps")]
use std::{collections::HashMap, io::Cursor};

use anyhow::Ok;
use cgmath::Vector4;
#[cfg(feature = "colormaps")]
use include_dir::Dir;
#[cfg(feature = "python")]
use numpy::ndarray::{ArrayViewD, Axis};
#[cfg(feature = "python")]
use pyo3::{pymethods, PyResult};

#[cfg(feature = "colormaps")]
use once_cell::sync::Lazy;

#[cfg(feature = "colormaps")]
use include_dir::include_dir;

pub const COLORMAP_RESOLUTION: u32 = 256;

#[cfg(feature = "colormaps")]
static COLORMAPS_MATPLOTLIB: include_dir::Dir = include_dir!("colormaps/matplotlib");
#[cfg(feature = "colormaps")]
static COLORMAPS_SEABORN: include_dir::Dir = include_dir!("colormaps/seaborn");
#[cfg(feature = "colormaps")]
static COLORMAPS_CMASHER: include_dir::Dir = include_dir!("colormaps/cmasher");
#[cfg(feature = "colormaps")]
static COLORMAPS_CUSTOM: include_dir::Dir = include_dir!("colormaps/custom");

#[cfg(feature = "colormaps")]
fn load_cmaps(dir: &Dir) -> HashMap<String, ColorMap> {
    let cmaps: HashMap<String, ColorMap> = dir
        .files()
        .filter_map(|f| {
            let file_name = f.path();
            let reader = Cursor::new(f.contents());
            let name = file_name.file_stem().unwrap().to_str().unwrap().to_string();
            let cmap = ColorMap::read(reader).unwrap();
            return Some((name, cmap));
        })
        .collect();
    cmaps
}

// list of predefined colormaps
#[cfg(feature = "colormaps")]
pub static COLORMAPS: Lazy<HashMap<String, HashMap<String, ColorMap>>> = Lazy::new(|| {
    let mut cmaps = HashMap::new();
    cmaps.insert("matplotlib".to_string(), load_cmaps(&COLORMAPS_MATPLOTLIB));
    cmaps.insert("seaborn".to_string(), load_cmaps(&COLORMAPS_SEABORN));
    cmaps.insert("cmasher".to_string(), load_cmaps(&COLORMAPS_CMASHER));
    cmaps.insert("custom".to_string(), load_cmaps(&COLORMAPS_CUSTOM));

    cmaps
});

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
#[cfg_attr(feature = "python", pyo3::pyclass)]

pub struct ColorMap {
    /// x, y0,y1
    #[serde(alias = "red")]
    pub r: Vec<(f32, f32, f32)>,

    #[serde(alias = "green")]
    pub g: Vec<(f32, f32, f32)>,

    #[serde(alias = "blue")]
    pub b: Vec<(f32, f32, f32)>,

    #[serde(alias = "alpha")]
    pub a: Option<Vec<(f32, f32, f32)>>,
}

impl ColorMap {
    pub fn new(
        r: Vec<(f32, f32, f32)>,
        g: Vec<(f32, f32, f32)>,
        b: Vec<(f32, f32, f32)>,
        a: Option<Vec<(f32, f32, f32)>>,
    ) -> anyhow::Result<Self> {
        if !Self::check_values(&r) {
            return Err(anyhow::anyhow!(
                "x values for red are not in (0,1) or ascending"
            ));
        };

        if !Self::check_values(&g) {
            return Err(anyhow::anyhow!(
                "x values for green are not in (0,1) or ascending"
            ));
        };

        if !Self::check_values(&b) {
            return Err(anyhow::anyhow!(
                "x values for blue are not in (0,1) or ascending"
            ));
        };

        if let Some(a) = &a {
            if !Self::check_values(&a) {
                return Err(anyhow::anyhow!(
                    "x values for alpha are not in (0,1) or ascending"
                ));
            };
        }
        Ok(Self { r, g, b, a })
    }

    pub fn from_json<R: Read>(reader: R) -> anyhow::Result<Self> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn check_values(v: &Vec<(f32, f32, f32)>) -> bool {
        let mut last_x = 0.0;
        for (x, _, _) in v.iter() {
            if x < &last_x || x > &1.0 || x < &0.0 {
                return false;
            }
            last_x = *x;
        }
        return true;
    }

    pub fn from_listed_colormap(colors: &Vec<Vector4<f32>>) -> Self {
        let mut r = vec![];
        let mut g = vec![];
        let mut b = vec![];
        let mut a = vec![];
        let n = colors.len();
        for (i, v) in colors.iter().enumerate() {
            let x = i as f32 / (n - 1) as f32;
            r.push((x, v.x, v.x));
            g.push((x, v.y, v.y));
            b.push((x, v.z, v.z));
            a.push((x, v.w, v.w));
        }

        // merge neighboring points with the same alpha value
        merge_neighbors(&mut r);
        merge_neighbors(&mut g);
        merge_neighbors(&mut b);
        merge_neighbors(&mut a);

        Self {
            r,
            g,
            b,
            a: Some(a),
        }
    }

    pub fn empty() -> Self {
        Self::new(
            vec![(0., 0., 0.), (1., 0., 0.)],
            vec![(0., 0., 0.), (1., 0., 0.)],
            vec![(0., 0., 0.), (1., 0., 0.)],
            None,
        )
        .unwrap()
    }

    fn sample(&self, x: f32) -> Vector4<u8> {
        let a = self
            .a
            .as_ref()
            .map(|a| sample_channel(x, &a))
            .unwrap_or(1.0);
        Vector4::new(
            (sample_channel(x, &self.r) * 255.) as u8,
            (sample_channel(x, &self.g) * 255.) as u8,
            (sample_channel(x, &self.b) * 255.) as u8,
            (a * 255.) as u8,
        )
    }

    pub fn reverse(&mut self) {
        let mut r: Vec<_> = self
            .r
            .iter()
            .map(|(x, y1, y2)| (1.0 - x, *y1, *y2))
            .collect();
        let mut g: Vec<_> = self
            .g
            .iter()
            .map(|(x, y1, y2)| (1.0 - x, *y1, *y2))
            .collect();
        let mut b: Vec<_> = self
            .b
            .iter()
            .map(|(x, y1, y2)| (1.0 - x, *y1, *y2))
            .collect();
        let mut a: Option<Vec<(f32, f32, f32)>> = self
            .a
            .clone()
            .map(|a| a.iter().map(|(x, y1, y2)| (1.0 - x, *y1, *y2)).collect());
        r.reverse();
        g.reverse();
        b.reverse();
        if let Some(a) = &mut a {
            a.reverse();
        }
        self.r = r;
        self.g = g;
        self.b = b;
        self.a = a;
    }

    pub fn read<R: Read + Seek>(mut reader: R) -> anyhow::Result<Self> {
        let mut start = [0; 6];
        reader.read_exact(&mut start)?;
        reader.seek(SeekFrom::Start(0))?;
        if start.eq(b"\x93NUMPY") {
            // numpy file
            Ok(Self::from_npy(reader)?)
        } else {
            // json file
            Ok(Self::from_json(reader)?)
        }
    }

    /// if all alpha values are 1.0, the alpha channel is considered boring
    #[allow(unused)]
    pub(crate) fn has_boring_alpha_channel(&self) -> bool {
        self.a
            .as_ref()
            .map(|a| a.iter().all(|(_, _, a)| *a == 1.0))
            .unwrap_or(true)
    }

    #[cfg(feature = "python")]
    pub fn from_array(data: ArrayViewD<f32>) -> Self {
        Self(
            data.axis_iter(Axis(0))
                .map(|v| {
                    Vector4::new(
                        (v[0] * 255.) as u8,
                        (v[1] * 255.) as u8,
                        (v[2] * 255.) as u8,
                        (v[3] * 255.) as u8,
                    )
                })
                .collect(),
        )
    }

    pub fn from_npy<'a, R>(reader: R) -> anyhow::Result<Self>
    where
        R: Read + Seek,
    {
        let npz_file = npyz::NpyFile::new(reader)?;
        let values: Vec<_> = npz_file
            .into_vec::<f32>()?
            .chunks_exact(4)
            .map(|v| Vector4::new(v[0], v[1], v[2], v[3]))
            .collect();
        Ok(Self::from_listed_colormap(&values))
    }

    pub fn rasterize(&self, n: usize) -> Vec<Vector4<u8>> {
        (0..n)
            .map(|i| self.sample(i as f32 / (n - 1) as f32))
            .collect()
    }

    pub(crate) fn from_color(color: Vector4<f32>, n: u32) -> ColorMap {
        Self::from_listed_colormap(&vec![color; n as usize])
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl ColorMap {
    #[new]
    fn __new__(
        r: Vec<(f32, f32, f32)>,
        g: Vec<(f32, f32, f32)>,
        b: Vec<(f32, f32, f32)>,
        a: Option<Vec<(f32, f32, f32)>>,
    ) -> PyResult<Self> {
        Self::new(r, g, b, a).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl Hash for ColorMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for c in [&self.r, &self.g, &self.b].iter() {
            c.iter().for_each(|(a, b, c)| {
                state.write_u32(a.to_bits());
                state.write_u32(b.to_bits());
                state.write_u32(c.to_bits())
            });
        }
        if let Some(a) = &self.a {
            a.iter().for_each(|(a, b, c)| {
                state.write_u32(a.to_bits());
                state.write_u32(b.to_bits());
                state.write_u32(c.to_bits())
            });
        }
    }
}

// samples a spline described by a list of points (x,y0,y1)
fn sample_channel(x: f32, values: &[(f32, f32, f32)]) -> f32 {
    for i in 0..values.len() - 1 {
        let (x0, _, y0) = values[i];
        let (x1, y1, _) = values[i + 1];
        if x0 <= x && x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    return 0.0;
}

/// removes points in a spline with the same y values
/// list contains points (x,y0,y1) where y0 ist the value before x and y1 the value after
fn merge_neighbors(values: &mut Vec<(f32, f32, f32)>) {
    let mut i = 1;
    while i < values.len() - 1 {
        let (_, y0, y1) = values[i];
        if y0 == y1 {
            let y_prev = values[i - 1].2;
            let y_next = values[i + 1].1;
            if y_prev == y_next {
                values.remove(i);
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}
