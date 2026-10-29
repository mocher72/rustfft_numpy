use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, Mutex, OnceLock};

static PLANS32: OnceLock<Mutex<std::collections::HashMap<(usize, bool), Arc<dyn Fft<f32>>>>> =
    OnceLock::new();
static PLANS64: OnceLock<Mutex<std::collections::HashMap<(usize, bool), Arc<dyn Fft<f64>>>>> =
    OnceLock::new();

fn plan32(n: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    let cache = PLANS32.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut plans = cache.lock().expect("RustFFT plan cache poisoned");
    plans
        .entry((n, inverse))
        .or_insert_with(|| {
            let mut planner = FftPlanner::<f32>::new();
            if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) }
        })
        .clone()
}

fn plan64(n: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    let cache = PLANS64.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut plans = cache.lock().expect("RustFFT plan cache poisoned");
    plans
        .entry((n, inverse))
        .or_insert_with(|| {
            let mut planner = FftPlanner::<f64>::new();
            if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) }
        })
        .clone()
}

#[pyfunction]
fn fft32<'py>(
    py: Python<'py>, input: PyReadonlyArray2<'py, Complex32>, inverse: bool,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let mut data = input.as_array().as_standard_layout().to_owned();
    let n = data.ncols();
    if n == 0 { return Err(PyValueError::new_err("FFT length must be greater than zero")); }
    let fft = plan32(n, inverse);
    let mut scratch = vec![Complex32::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    for mut row in data.rows_mut() {
        fft.process_with_scratch(row.as_slice_mut().expect("standard-layout row"), &mut scratch);
    }
    Ok(data.into_pyarray(py))
}

#[pyfunction]
fn fft64<'py>(
    py: Python<'py>, input: PyReadonlyArray2<'py, Complex64>, inverse: bool,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let mut data = input.as_array().as_standard_layout().to_owned();
    let n = data.ncols();
    if n == 0 { return Err(PyValueError::new_err("FFT length must be greater than zero")); }
    let fft = plan64(n, inverse);
    let mut scratch = vec![Complex64::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    for mut row in data.rows_mut() {
        fft.process_with_scratch(row.as_slice_mut().expect("standard-layout row"), &mut scratch);
    }
    Ok(data.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft32, m)?)?;
    m.add_function(wrap_pyfunction!(fft64, m)?)?;
    Ok(())
}
