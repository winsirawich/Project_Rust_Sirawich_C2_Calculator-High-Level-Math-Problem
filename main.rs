use clap::{App, Arg};
use std::collections::HashSet;
use std::io;
use std::collections::HashMap;
extern crate clap;

// Rest of your code

// Define your struct and functions here
#[derive(Debug, Clone)]
struct MathSet {
    elements: HashSet<i32>,
}

impl MathSet {
    // Create a new empty MathSet
    fn new() -> Self {
        MathSet {
            elements: HashSet::new(),
        }
    }
    // Add an element to the set
    fn add(&mut self, element: i32) {
        self.elements.insert(element);
    }

    // Remove an element from the set
    fn remove(&mut self, element: i32) {
        self.elements.remove(&element);
    }

    // Check if an element is in the set
    fn contains(&self, element: i32) -> bool {
        self.elements.contains(&element)
    }

    // Compute the union of two sets
    fn union(&self, other: &MathSet) -> MathSet {
        let mut result = self.clone();
        for &element in &other.elements {
            result.add(element);
        }
        result
    }

    // Compute the intersection of two sets
    fn intersection(&self, other: &MathSet) -> MathSet {
        let mut result = MathSet::new();
        for &element in &self.elements {
            if other.contains(element) {
                result.add(element);
            }
        }
        result
    }

    // Compute the difference of two sets (self - other)
    fn difference(&self, other: &MathSet) -> MathSet {
        let mut result = self.clone();
        for &element in &other.elements {
            result.remove(element);
        }
        result
    }
}

fn read_input(set: &mut MathSet) {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    let elements: Vec<i32> = input
        .split_whitespace()
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    for element in elements {
        set.add(element);
    }
}

fn read_matrix(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; columns]; rows];
    println!("Enter matrix elements ({} rows x {} columns):", rows, columns);
    for i in 0..rows {
        for j in 0..columns {
            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("Failed to read line");
            let element: f64 = input.trim().parse().expect("Invalid input");
            matrix[i][j] = element;
        }
    }
    matrix
}

fn display_matrix(matrix: &[Vec<f64>]) {
    for row in matrix {
        for element in row {
            print!("{:.2}\t", element);
        }
        println!();
    }
}

fn matrix_addition(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix1.len();
    let columns = matrix1[0].len();
    let mut result = vec![vec![0.0; columns]; rows];

    for i in 0..rows {
        for j in 0..columns {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    result
}

fn matrix_subtraction(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix1.len();
    let columns = matrix1[0].len();
    let mut result = vec![vec![0.0; columns]; rows];

    for i in 0..rows {
        for j in 0..columns {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    result
}

fn matrix_multiplication(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows1 = matrix1.len();
    let columns1 = matrix1[0].len();
    let rows2 = matrix2.len();
    let columns2 = matrix2[0].len();

    assert!(columns1 == rows2, "Matrix dimensions mismatch for multiplication");

    let mut result = vec![vec![0.0; columns2]; rows1];

    for i in 0..rows1 {
        for j in 0..columns2 {
            for k in 0..columns1 {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    result
}

fn matrix_division(matrix: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let columns = matrix[0].len();
    let mut result = vec![vec![0.0; columns]; rows];

    for i in 0..rows {
        for j in 0..columns {
            result[i][j] = matrix[i][j] / scalar;
        }
    }

    result
}

fn calculate_mean(numbers: &Vec<i32>) -> f64 {
    let sum: i32 = numbers.iter().sum();
    let count = numbers.len() as f64;
    sum as f64 / count
}

fn calculate_median(numbers: &Vec<i32>) -> f64 {
    let mut sorted_numbers = numbers.clone();
    sorted_numbers.sort();
    let mid = sorted_numbers.len() / 2;
    if sorted_numbers.len() % 2 == 0 {
        let middle_values = &sorted_numbers[mid - 1..=mid];
        let sum: i32 = middle_values.iter().sum();
        sum as f64 / 2.0
    } else {
        sorted_numbers[mid] as f64
    }
}

fn calculate_mode(numbers: &Vec<i32>) -> Vec<i32> {
    let mut counts = HashMap::new();
    let mut max_count = 0;

    for &num in numbers {
        let count = counts.entry(num).or_insert(0);
        *count += 1;
        if *count > max_count {
            max_count = *count;
        }
    }

    let mode_values: Vec<i32> = counts
        .into_iter()
        .filter(|&(_, count)| count == max_count)
        .map(|(value, _)| value)
        .collect();

    mode_values
}

fn calculate_sum(numbers: &Vec<i32>) -> i32 {
    numbers.iter().sum()
}

fn and_operation() -> bool {
    let input1 = read_single_input();
    let input2 = read_single_input();
    input1 && input2
}

fn or_operation() -> bool {
    let input1 = read_single_input();
    let input2 = read_single_input();
    input1 || input2
}


fn not_operation() -> bool {
    let input = read_single_input();
    !input
}

fn read_inputs() -> (bool, bool) {
    println!("Enter two Boolean values (true/false):");
    let input1 = read_single_input();
    let input2 = read_single_input();
    (input1, input2)
}

fn read_single_input() -> bool {
    loop {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        match input.trim().to_lowercase().as_str() {
            "true" => return true,
            "false" => return false,
            _ => println!("Invalid input. Please enter 'true' or 'false'."),
        }
    }
}

// Define a struct to represent a Complex Number
#[derive(Debug)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    // Create a new Complex Number
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    // Add two Complex Numbers
    fn add(&self, other: &Complex) -> Complex {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }

    // Subtract two Complex Numbers
    fn subtract(&self, other: &Complex) -> Complex {
        Complex::new(self.real - other.real, self.imag - other.imag)
    }

    // Multiply two Complex Numbers
    fn multiply(&self, other: &Complex) -> Complex {
        let real_part = (self.real * other.real) - (self.imag * other.imag);
        let imag_part = (self.real * other.imag) + (self.imag * other.real);
        Complex::new(real_part, imag_part)
    }

    // Calculate the magnitude (absolute value) of a Complex Number
    fn magnitude(&self) -> f64 {
        (self.real.powi(2) + self.imag.powi(2)).sqrt()
    }

    // Display a Complex Number as a string
    fn to_string(&self) -> String {
        format!("{} + {}i", self.real, self.imag)
    }
}

fn read_complex_numbers() -> (Complex, Complex) {
    println!("Enter the real and imaginary parts of the first complex number (e.g., 3 4):");
    let c1 = read_single_complex_number();

    println!("Enter the real and imaginary parts of the second complex number:");
    let c2 = read_single_complex_number();

    (c1, c2)
}

fn read_single_complex_number() -> Complex {
    loop {
        println!("Enter a complex number (e.g., 3 4 for 3 + 4i):");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let parts: Vec<f64> = input
            .split_whitespace()
            .map(|s| s.parse().expect("Invalid input"))
            .collect();

        if parts.len() == 2 {
            return Complex::new(parts[0], parts[1]);
        } else {
            println!("Invalid input. Please enter a valid complex number.");
        }
    }
}

fn main() {
    let matches = App::new("Calculator")
        .version("1.0")
        .author("You")
        .about("Solving various high school level Math problems.
        ")
        .subcommand(
            App::new("set")
                .about("Performs set operations")
                // Add your set operation subcommands and arguments here using clap
        )
        .subcommand(
            App::new("matrix")
                .about("Performs matrix operations")
                .subcommand(
                    App::new("add")
                        .about("Adds two matrices")
                        // Add arguments specific to matrix addition
                )
                .subcommand(
                    App::new("subtract")
                        .about("Subtracts one matrix from another")
                        // Add arguments specific to matrix subtraction
                )
                .subcommand(
                    App::new("multiply")
                        .about("Multiplies two matrices")
                        // Add arguments specific to matrix multiplication
                )
                .subcommand(
                    App::new("divide")
                        .about("Divides a matrix by a scalar value")
                        // Add arguments specific to matrix division
                )
        )
        .subcommand(
            App::new("vector")
                .about("Performs vector operations")
                // Add your vector operation subcommands and arguments here using clap
        )
        .subcommand(
            App::new("boolean")
                .about("Performs boolean logic operations")
                // Add your boolean operation subcommands and arguments here using clap
        )
        .subcommand(
            App::new("complex")
                .about("Performs complex number operations")
                // Add your complex number operation subcommands and arguments here using clap
        )
        .get_matches();

        match matches.subcommand() {
            ("set", Some(_)) => {
                // Add elements to the sets
                let mut set1 = MathSet::new();
                let mut set2 = MathSet::new();
    
                println!("Enter elements for the first set (space-separated integers):");
                read_input(&mut set1);
    
                println!("Enter elements for the second set (space-separated integers):");
                read_input(&mut set2);
    
                // Perform set operations here
                let union_set = set1.union(&set2);
                let intersection_set = set1.intersection(&set2);
                let difference_set = set1.difference(&set2);
    
                // Example usage of the set operations
                println!("Union: {:?}", union_set);
                println!("Intersection: {:?}", intersection_set);
                println!("Difference: {:?}", difference_set);
            }
            ("matrix", Some(matrix_matches)) => {
                match matrix_matches.subcommand() {
                    ("add", Some(_)) => {
                        // Matrix Addition Operation
                        // TODO: Add your matrix addition code here
                        println!("Performing matrix addition");
                        // Example operations:
                        let matrix1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
                        let matrix2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
                        let result = matrix_addition(&matrix1, &matrix2);
                        println!("Matrix addition result: {:?}", result);
                    }
                    ("subtract", Some(_)) => {
                        // Matrix Subtraction Operation
                        // TODO: Add your matrix subtraction code here
                        println!("Performing matrix subtraction");
                        // Example operations:
                        let matrix1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
                        let matrix2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
                        let result = matrix_subtraction(&matrix1, &matrix2);
                        println!("Matrix subtraction result: {:?}", result);
                    }
                    ("multiply", Some(_)) => {
                        // Matrix Multiplication Operation
                        // TODO: Add your matrix multiplication code here
                        println!("Performing matrix multiplication");
                        // Example operations:
                        let matrix1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
                        let matrix2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
                        let result = matrix_multiplication(&matrix1, &matrix2);
                        println!("Matrix multiplication result: {:?}", result);
                    }
                    ("divide", Some(_)) => {
                        // Matrix Division Operation
                        // TODO: Add your matrix division code here
                        println!("Performing matrix division");
                        // Example operations:
                        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
                        let scalar = 2.0;
                        let result = matrix_division(&matrix, scalar);
                        println!("Matrix division result: {:?}", result);
                    }
                    _ => {}
                }
            }
            ("vector", Some(_)) => {
                // Vector Operations
                println!("Enter elements for the vector (space-separated integers):");
                let mut input = String::new();
                io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");
    
                let numbers: Vec<i32> = input
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
    
                // Example operations:
                let mean = calculate_mean(&numbers);
                let median = calculate_median(&numbers);
                let mode = calculate_mode(&numbers);
                let sum = calculate_sum(&numbers);
                println!("Mean: {}, Median: {}, Mode: {:?}, Sum: {}", mean, median, mode, sum);
            }
            ("boolean", Some(_)) => {
                // Boolean Logic Operations
                // TODO: Add your boolean logic operations code here
                println!("Performing boolean logic operations");
                // Example operations:
                let result_and = and_operation();
                let result_or = or_operation();
                let result_not = not_operation();
                println!("AND result: {}, OR result: {}, NOT result: {}", result_and, result_or, result_not);
            }
            ("complex", Some(_)) => {
                // Complex Number Operations
                // TODO: Add your complex number operations code here
                println!("Performing complex number operations");
                // Example operations:
                let (c1, c2) = read_complex_numbers();
                let result_add = c1.add(&c2);
                let result_subtract = c1.subtract(&c2);
                let result_multiply = c1.multiply(&c2);
                let magnitude = c1.magnitude();
                println!("Add result: {:?}, Subtract result: {:?}, Multiply result: {:?}, Magnitude: {}", result_add, result_subtract, result_multiply, magnitude);
            }
            _ => {}
        }
    }





    