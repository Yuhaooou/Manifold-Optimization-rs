use ndarray::Array2;

use super::lapack::LapackElem;

pub trait LinalgQR {
    type Elem;

    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>);
}

impl<T> LinalgQR for Array2<T>
where
    T: LapackElem,
{
    type Elem = T;
    fn qr_(&self) -> (Array2<Self::Elem>, Array2<Self::Elem>) {
        todo!()
    }
}
