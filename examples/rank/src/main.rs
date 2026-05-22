use manifold_optimization::{
    algorithm::RTR,
    function::FuncGradHess,
    manifolds::{
        RandomPoint,
        fixed_rank::{FixedRank, Point, TangentVector},
    },
    problem::Problem,
    utils::traits::Norm,
};
use ndarray::{Zip, prelude::*};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use rand::distr::Bernoulli;

fn hessian(x: &Point<f64>, v: &TangentVector<f64>, sample: &Array2<bool>) -> Array2<f64> {
    let mut res = v.full(x);
    Zip::from(&mut res).and(sample).par_for_each(|res, &s| {
        if !s {
            *res = 0.;
        }
    });
    res
}

fn compute(
    x: &Point<f64>,
    v: Option<&TangentVector<f64>>,
    mat_a: &Point<f64>,
    sample: &Array2<bool>,
    compute_value: bool,
    compute_grad: bool,
    compute_hess: bool,
) -> (Option<f64>, Option<Array2<f64>>, Option<Array2<f64>>) {
    let (m, n) = sample.dim();

    let (value, grad) = if compute_value || compute_grad {
        let l1 = x.u() * x.s();
        let l2 = mat_a.u() * mat_a.s();
        let r1 = x.v();
        let r2 = mat_a.v();

        let mut value = if compute_value { Some(0.) } else { None };

        let mut grad = if compute_grad {
            Some(Array::zeros((m, n)))
        } else {
            None
        };

        for i in 0..m {
            for j in 0..n {
                if sample[[i, j]] {
                    let tmp = l1.row(i).dot(&r1.row(j)) - l2.row(i).dot(&r2.row(j));
                    if compute_value {
                        value = Some(value.unwrap() + tmp.powi(2));
                    }
                    if compute_grad {
                        grad.as_mut().unwrap()[[i, j]] = tmp;
                    }
                }
            }
        }

        let value = if compute_value {
            Some(value.unwrap() * 0.5)
        } else {
            None
        };

        (value, grad)
    } else {
        (None, None)
    };

    let hess = if compute_hess {
        Some(hessian(x, v.unwrap(), sample))
    } else {
        None
    };
    (value, grad, hess)
}

fn main() {
    let m = 500;
    let n = 300;
    let r = 40;
    let sample_rate = 0.5;

    let mut rng = rand::rng();
    let dist = Normal::new(0., 1.).unwrap();
    let mat_al = Array2::random_using((m, r), dist.clone(), &mut rng);
    let mat_ar = Array2::random_using((n, r), dist.clone(), &mut rng);
    let mat_a = Point::new_from_full(&mat_al.dot(&mat_ar.t()), r);

    let sample = Array2::random_using((m, n), Bernoulli::new(sample_rate).unwrap(), &mut rng);

    let manifold = FixedRank::<f64>::new(m, n, r);
    let init_point = manifold.random_point_with_rng(&mut rng);

    let start_err = (init_point.full() - mat_a.full()).norm() / mat_a.full().norm();

    let function = FuncGradHess::new_from_ambient(manifold, |x, v, c1, c2, c3| {
        compute(x, v, &mat_a, &sample, c1, c2, c3)
    });
    let problem = Problem::new(function);

    let mut rtr = RTR::new(problem, 1000., 0.1)
        .set_max_iterations(1000)
        .set_verbose(1)
        .set_min_grad_norm(1e-6)
        .set_min_step_size(1e-9);

    let res = rtr.run(10.);

    let final_point = res.point.full();
    let a_full = mat_a.full();
    let err = (final_point - &a_full).norm() / a_full.norm();

    println!("start error: {:.6}", start_err);
    println!("end error: {:.4e}", err);
}
