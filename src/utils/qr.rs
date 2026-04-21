use ndarray::Array2;
use ndarray_linalg::QR;
use num_complex::{Complex32 as c32, Complex64 as c64};

use crate::utils::LinalgBase;

pub trait LinalgQR: LinalgBase {
    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>);
}

impl LinalgQR for Array2<f64> {
    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.qr().unwrap()
    }
}

impl LinalgQR for Array2<f32> {
    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.qr().unwrap()
    }
}

impl LinalgQR for Array2<c64> {
    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.qr().unwrap()
    }
}

impl LinalgQR for Array2<c32> {
    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        self.qr().unwrap()
    }
}
