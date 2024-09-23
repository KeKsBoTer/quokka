use cgmath::Vector2;
use half::f16;
use image::{ImageBuffer, Rgba};
use numpy::{ndarray::StrideShape, IntoPyArray, PyArray4, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::env::{self};

use crate::{cmap::LinearSegmentedColorMap, offline::render_volume, presets::Preset, renderer::{DVRSettings, IsoSettings, RenderSettings, SSAOSettings}, viewer, volume::Volume};

#[pymodule]
fn quokka<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    fn render_video<'py>(
        py: Python<'py>,
        volume: PyReadonlyArrayDyn<'py, f16>,
        width: u32,
        height: u32,
        time: Vec<f32>,
        preset: Preset,
    ) -> Bound<'py, PyArray4<u8>> {
        let volume = Volume::from_array(volume.as_array().to_owned()).expect("cannot read array");
        let img: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = pollster::block_on(render_volume(
            vec![volume],
            Vector2::new(width, height),
            &time,
            preset,
        ))
        .unwrap();

        let shape = StrideShape::from((time.len(), width as usize, height as usize, 4 as usize));
        let arr = numpy::ndarray::Array4::from_shape_vec(
            shape,
            img.iter().flat_map(|img| img.to_vec()).collect(),
        )
        .unwrap();
        return arr.into_pyarray_bound(py);
    }

    #[pyfn(m)]
    fn standalone<'py>(_py: Python<'py>) -> PyResult<()> {
        // donts pass first argument (binary name) to parser
        match pollster::block_on(viewer(env::args().skip(1))) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "{:?}",
                e
            ))),
        }
    }

    m.add_class::<Preset>()?;
    m.add_class::<RenderSettings>()?;
    m.add_class::<DVRSettings>()?;
    m.add_class::<IsoSettings>()?;
    m.add_class::<SSAOSettings>()?;
    m.add_class::<LinearSegmentedColorMap>()?;
    
    Ok(())
}
