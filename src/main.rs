use num::complex::Complex;
use std::f64::consts::PI;
use std::iter::successors;

#[derive(Debug)]
struct Polynomial {
    coeffs: Vec<Complex<f64>>,
}

impl From<Vec<f64>> for Polynomial {
    fn from(input: Vec<f64>) -> Self {
        Polynomial { coeffs: input.into_iter().map(Complex::from).collect() }
    }
}

impl std::fmt::Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (exp, c) in self.coeffs.iter().enumerate().rev() {
            write!(f, "{:+.2}*x^{} ", c.re, exp)?;
        }

        Ok(())
    }
}

impl Polynomial {
    fn fft(input: &mut [Complex<f64>]) -> Vec<Complex<f64>> {
        Polynomial::evaluate(input, 1.)
    }

    fn inverse_fft(input: &mut [Complex<f64>]) -> Vec<Complex<f64>> {
        let mut coeffs = Polynomial::evaluate(input, -1.);
        let n = coeffs.len() as f64;
        coeffs.iter_mut().for_each(|x| *x /= n);
        coeffs
    }

    fn shuffle_coeffs(input: &mut [Complex<f64>], n: usize, log: u32) {
        (0..n)
            .map(|x| (x, x.reverse_bits() >> (8 * std::mem::size_of_val(&n) as u32 - log)))
            .filter(|(a, b)| a < b)
            .for_each(|(i, j)| input.swap(i, j));
    }

    fn mul(&mut self, rhs: &mut Self) -> Polynomial {
        let n = self.coeffs.len();
        let m = rhs.coeffs.len();

        let new_size = n + m - 1;
        let new_aligned_size = new_size.next_power_of_two();
        self.coeffs.resize_with(new_aligned_size, Default::default);
        rhs.coeffs.resize_with(new_aligned_size, Default::default);

        let self_points = Polynomial::fft(&mut self.coeffs);
        let rhs_points = Polynomial::fft(&mut rhs.coeffs);

        let mut new_points: Vec<Complex<f64>> =
            self_points.iter().zip(rhs_points.iter()).map(|(x, y)| x * y).collect();

        let mut new_coeffs = Polynomial::inverse_fft(&mut new_points);

        self.coeffs.truncate(n);
        rhs.coeffs.truncate(m);
        new_coeffs.truncate(new_size);

        Polynomial { coeffs: new_coeffs }
    }

    fn evaluate(input: &mut [Complex<f64>], sign: f64) -> Vec<Complex<f64>> {
        let n = input.len();

        assert!(n.is_power_of_two());
        let log = n.trailing_zeros();

        Polynomial::shuffle_coeffs(input, n, log);

        let mut ret: Vec<Complex<f64>> = input.to_owned();

        for step in (1..).map(|x| 1 << x).take(log as usize) {
            let part = sign * 2. * PI / step as f64;
            let first: Complex<f64> = Complex::new(part.cos(), part.sin());
            let iter =
                successors(Some(Complex::new(1., 0.)), |prev| Some(prev * first)).take(n / 2);

            for i in (0..n).step_by(step) {
                for (num, j) in iter.clone().zip(0..step / 2) {
                    let a = ret[i + j];
                    let b = num * ret[i + j + step / 2];

                    ret[i + j] = a + b;
                    ret[i + j + step / 2] = a - b;
                }
            }
        }

        Polynomial::shuffle_coeffs(input, n, log);

        ret
    }
}

fn main() {
    let mut a: Polynomial = vec![7., -1., 4., 3.].into();
    let mut b: Polynomial = vec![3., -2., -4., 7.].into();

    println!("a = {},\nb = {}\n", a, b);
    // println!("a = {},\n", a);

    // println!("{:?}", a.coeffs);
    // let mut a_points = Polynomial::fft(&mut a.coeffs);
    // println!("{:?}", a_points);
    // let a_coeffs = Polynomial::inverse_fft(&mut a_points);
    // println!("{:?}", a_coeffs);

    println!("{}", a.mul(&mut b));
}
