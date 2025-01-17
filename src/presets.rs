use std::{collections::HashMap, io::Cursor};

use cgmath::Point3;
use include_dir::include_dir;
use include_dir::Dir;
use once_cell::sync::Lazy;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::renderer::RenderSettings;

static PRESETS_FOLDER: include_dir::Dir = include_dir!("presets");

fn load_presets(dir: &Dir) -> HashMap<String, Preset> {
    let cmaps: HashMap<String, Preset> = dir
        .files()
        .filter_map(|f| {
            let file_name = f.path();
            let reader = Cursor::new(f.contents());
            let name = file_name.file_stem().unwrap().to_str().unwrap().to_string();
            let cmap = Preset::from_json(reader);
            match cmap {
                Ok(cmap) => {
                    log::debug!("Loaded preset {}", name);
                    Some((name, cmap))
                }
                Err(e) => {
                    log::warn!("Failed to load preset {}: {}", file_name.display(), e);
                    None
                }
            }
        })
        .collect();
    cmaps
}

// list of predefined presets
pub static PRESETS: once_cell::sync::Lazy<HashMap<String, Preset>> =
    Lazy::new(|| load_presets(&PRESETS_FOLDER));

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct Preset {
    pub name: String,
    pub render_settings: RenderSettings,
    pub camera: Option<Point3<f32>>,
}

impl Preset {
    pub fn from_json<R: std::io::Read>(reader: R) -> anyhow::Result<Self> {
        Ok(serde_json::from_reader(reader)?)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Preset {
    #[new]
    fn __new__(
        name: String,
        render_settings: RenderSettings,
        cmap: Option<ColorMap>,
        camera: Option<(f32, f32, f32)>,
    ) -> Self {
        Preset {
            name,
            render_settings,
            cmap,
            camera: camera.map(|(x, y, z)| Point3::new(x, y, z)),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Presets(pub HashMap<String, Preset>);

impl Presets {
    #[cfg(target_arch = "wasm32")]
    pub fn from_local_storage() -> anyhow::Result<Self> {
        use crate::local_storage;

        let local_storage = local_storage()?;
        let presets = local_storage.get("presets").map_err(|e| {
            anyhow::anyhow!(e
                .as_string()
                .unwrap_or("failed to load presets from local storage".to_string()))
        })?;
        match presets {
            Some(presets) => Ok(serde_json::from_str(&presets)?),
            None => Ok(Self(HashMap::new())),
        }
    }
}
