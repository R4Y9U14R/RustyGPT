extern crate libc;

use libc::{c_float, c_int};

extern "C" {
    fn check_is_cuda_available() -> bool;
    fn launch_add_cpu(A: *const c_float, B: *const c_float, result: *mut c_float, num_rows: c_int, num_cols: c_int);
    fn launch_add(A: *const c_float, B: *const c_float, result: *mut c_float, num_rows: c_int, num_cols: c_int);
    fn launch_matmul_cpu(A: *const c_float, B: *const c_float, result: *mut c_float, m_A: c_int, n_A: c_int, m_B: c_int, n_B: c_int);
    fn launch_matmul(A: *const c_float, B: *const c_float, result: *mut c_float, m_A: c_int, n_A: c_int, m_B: c_int, n_B: c_int);
    fn launch_transpose_cpu(A: *const c_float, result: *mut c_float, num_rows: c_int, num_cols: c_int);
    fn launch_transpose(A: *const c_float, result: *mut c_float, num_rows: c_int, num_cols: c_int);

}

pub fn is_cuda_available() -> bool {
    let is_available;
    unsafe {
        is_available = check_is_cuda_available();
    }
    is_available
}

pub fn format_as_matrix(arr: &[f32], dimx: usize, dimy: usize) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(dimx);
    for i in 0..dimx {
        let start = i * dimy;
        let end = (i + 1) * dimy;
        result.push(arr[start..end].to_vec());
    }
    result
}

pub fn add_cpu(a: &[f32], b: &[f32], num_rows: i32, num_cols: i32) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; (num_rows * num_cols) as usize];
    unsafe {
        launch_add_cpu(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), num_rows, num_cols);
    }
    result
}

pub fn add(a: &[f32], b: &[f32], num_rows: i32, num_cols: i32) -> Vec<f32> {
    let mut result: Vec<f32> =vec![0.0; (num_rows * num_cols) as usize];
    unsafe {
        launch_add(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), num_rows, num_cols);
    }
    result
}

pub fn matmul_cpu(a: &[f32], b: &[f32], m_a: i32, n_a: i32, m_b: i32, n_b: i32) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; (m_a * n_b) as usize];
    unsafe {
        launch_matmul_cpu(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), m_a, n_a, m_b, n_b);
    }
    result
}

pub fn matmul(a: &[f32], b: &[f32], m_a: i32, n_a: i32, m_b: i32, n_b: i32) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; (m_a * n_b) as usize];
    unsafe {
        launch_matmul(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), m_a, n_a, m_b, n_b);
    }
    result
}

pub fn transpose_cpu(a: &[f32], num_rows: i32, num_cols: i32) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; (num_rows * num_cols) as usize];
    unsafe {
        launch_transpose_cpu(a.as_ptr(), result.as_mut_ptr(), num_rows, num_cols);
    }
    result
}

pub fn transpose(a: &[f32], num_rows: i32, num_cols: i32) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; (num_rows * num_cols) as usize];
    unsafe {
        launch_transpose(a.as_ptr(), result.as_mut_ptr(), num_rows, num_cols);
    }
    result
}
