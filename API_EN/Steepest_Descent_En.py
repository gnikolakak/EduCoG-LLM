# The code creates the Steepest Descent algorithm to find the minimum of a function of two variables
# The user enters the initial points, parameters and the function he wanted to examine
# The algorithm calculates the derivatives and the slope of the function, updating the points at each step
# The process teminates:
# 1) when one of the termination criteria is satisfied (before the maximum number of iterations is exceeded), then the results are displayed on the user's screen
# 2) when the maximum number of iterations is exceeded, informing the user that the number of iterations has been exceeded without displaying the graphs

import sympy as sp 
import numpy as np 
import matplotlib.pyplot as plt 

def Initial_Point(): 
    while True: 
        try: 
            print("Initial Point & Learning Rate: ") 
            x0 = float(input("x0 is the coordinate of the x'x axis. Please enter x0: ")) 
            y0 = float(input("y0 is the coordinate of the y'y axis. Please enter y0: ")) 
            return x0, y0 

        except ValueError: 
            print("Invalid input! Please enter valid numbers for x0 and y0.") 

def Parameters(): 
    while True: 
        try: 
            a = float(input("a is learning rate. Please enter a: ")) 
            print("-------------------------------------------------------------------------------------")

            print("The Steepest Descent algorithm stops if one of the following termination criteria is satisfied: \n"
            "1. In case the number of iterations exceeds 1000 and any of the criteria has not been satisfied.\n"
            "2. The slope of the function at the point found is less than a constant c1 defined by the user. \n"
            "If the slope is equal to 0 at the starting point (x0,y0), then it prompts the user for a new starting point.\n"
            "3. The distance between two consecutive points must be less than a constant c2 defined by the user. \n"
            "4. The value of the function between two consecutive points must be less than a constant c3 defined by the user. \n")
            print("-------------------------------------------------------------------------------------")

            print("Therefore, for the algorithm to proceed, c1, c2, c3 are required from the user:")
            c1 = float(input("1st) What is c1 so that the criterion |∇f| ≤ c1 can be examined? ")) 
            c2 = float(input("2nd) What is c2 so that the criterion |Xn - Xn-1| ≤ c2 can be examined? ")) 
            c3 = float(input("3rd) What is c3 so that the criterion |f(Xn) - f(Xn-1)| ≤ c3 can be examined? ")) 
            return a, c1, c2, c3 

        except ValueError: 
            print("Invalid input! Please enter valid numbers.") 

def f(): 
    while True: 
        try: 
            print("-------------------------------------------------------------------------------------")
            expr_str = input("Please enter the function in x and y form (e.g., x**2 + y**4): ") 
            expr = sp.sympify(expr_str) 
            return expr 

        except sp.SympifyError: 
            print("Invalid input! Please enter a valid function.") 
            
def steepest_descent(f_num, x0, y0, a, c1, c2, c3, derivative_x, derivative_y): 
    tries = 0 
    MAX_TRIES = 1000 
    points = [] 
    x, y = sp.symbols('x y') 
    
    x_path = [x0] 
    y_path = [y0] 
    z_path = [f_num(x0, y0)] 

    slope_x = derivative_x.subs({x: x0, y: y0}).evalf() 
    slope_y = derivative_y.subs({x: x0, y: y0}).evalf() 

    grad_norm = sp.sqrt(slope_x**2 + slope_y**2) 
    
    slope_x_expr = derivative_x 
    slope_y_expr = derivative_y 
    slope_x = slope_x_expr.subs({x: x0, y: y0}).evalf() 
    slope_y = slope_y_expr.subs({x: x0, y: y0}).evalf() 
    print("The partial derivative with respect to x is: ", slope_x_expr) 
    print("The partial derivative with respect to y is: ", slope_y_expr) 
    
    while grad_norm == 0:
        print("The slope at point (x0, y0) is 0. Please enter a new starting point.")
        x0, y0 = Initial_Point() 

        slope_x = derivative_x.subs({x: x0, y: y0}).evalf()
        slope_y = derivative_y.subs({x: x0, y: y0}).evalf()
        grad_norm = sp.sqrt(slope_x**2 + slope_y**2)

    while tries <= MAX_TRIES:
        slope_x = derivative_x.subs({x: x0, y: y0}).evalf()
        slope_y = derivative_y.subs({x: x0, y: y0}).evalf()
        grad_norm = sp.sqrt(slope_x**2 + slope_y**2)

        points.append((x0, y0))

        if grad_norm < c1: 
            criterion = "1st criterion: The slope is small." 
            break 

        elif len(points) > 1: 
            prev_x, prev_y = points[-2] 
            distance = sp.sqrt((x0 - prev_x)**2 + (y0 - prev_y)**2).evalf() 
            
            if distance < c2: 
                criterion = "2nd criterion: The distance between two consecutive points is small." 
                break  

            f_prev = f_num(prev_x, prev_y) 
            f_current = f_num(x0, y0) 
            
            if abs(f_current - f_prev) < c3: 
                criterion = "3rd criterion: The convergence of the function is small." 
                break

        x0 = x0 - a * slope_x
        y0 = y0 - a * slope_y

        x_path.append(x0) 
        y_path.append(y0) 
        z_path.append(f_num(x0, y0)) 
        
        tries += 1
    
    if tries > MAX_TRIES: 
        criterion = f"Search failed: maximum number of iterations reached ({MAX_TRIES})." 
        return x0, y0, None, criterion, False, x_path, y_path, z_path, tries 
        
    return x0, y0, f_num(x0, y0), criterion, True, x_path, y_path, z_path, tries

def main(): 
    x0, y0 = Initial_Point() 
    a, c1, c2, c3 = Parameters() 

    f_sym = f() 
    x, y = sp.symbols('x y') 

    f_num = sp.lambdify((x, y), f_sym, "numpy")

    derivative_x_sym = sp.diff(f_sym, x) 
    derivative_y_sym = sp.diff(f_sym, y)  

    min_x, min_y, min_value, criterion, show_plots, x_path, y_path, z_path, total_tries = steepest_descent(f_num, x0, y0, a, c1, c2, c3, derivative_x_sym, derivative_y_sym)

    if not show_plots:
        print("Maximum number of repetitions exceeded.") 
        return

    print("-------------------------------------------------------------------------------------")
    print(f"Minimum point: ({min_x}, {min_y}), with function value f(x, y) = {min_value}") 
    print("Convergence criterion ->", criterion) 
    print(f"Total repetitions: {total_tries}") 

    x_vals = np.linspace(-1.5, 1.5, 400) 
    y_vals = np.linspace(-1.5, 1.5, 400) 
    X, Y = np.meshgrid(x_vals, y_vals) 
    Z = f_num(X, Y) 
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis') 
    ax1.plot(x_path, y_path, z_path, 'r-', label='Steepest Descent Path') 
    ax1.scatter(min_x, min_y, min_value, color='red', s=100, label='Ελάχιστο Σημείο')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$'),
    ax1.set_zlabel('$f(x, y)$')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.contour(X, Y, Z, levels=50, cmap='viridis') 
    ax2.plot(x_path, y_path, 'r-', label='Steepest Descent Path')
    ax2.scatter(min_x, min_y, color='red', s=100, label='Ελάχιστο Σημείο') 
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.legend()

    plt.show()

main()
