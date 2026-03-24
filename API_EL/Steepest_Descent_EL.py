# Ο κώδικας υλοποιεί τον Αλγόριθμο της Steepest Descent για την εύρεση του ελαχίστου μιας συνάρτησης δύο μεταβλητών
# Ο χρήστης εισάγει τα αρχικά σημεία, τις παραμέτρους και την συνάρτηση που θα ήθελε να εξετάσει
# Ο αλγόριθμος υπολογίζει τις παραγώγους και την κλίση της συνάρτησης, ενημερώνοντας τα σημεία σε κάθε βήμα
# Η διαδικασία τερματίζεται:
# 1) όταν ικανοποιηθεί ένα από τα κριτήρια τερματισμού (προτού ξεπεραστεί ο μέγιστος αριθμός επαναλήψεων) και έτσι εμφανίζονται τα αποτελέσματα και τα γραφήματα (σε τρισδιάστατη και δισδιάστατη μορφή) που δείχνουν την πορεία του αλγορίθμου
# 2) όταν ξεπεραστεί ο μέγιστος αριθμός επαναλήψεων ενημερώνοντας τον χρήστη πως ο αριθμός των επαναλήψεων ξεπεράστηκε χωρίς να εμφανιστούν οι γραφικές παραστάσεις

import sympy as sp 
import numpy as np 
import matplotlib.pyplot as plt 

def Initial_Point(): 
    while True: 
        try: 
            print("Αρχικά σημεία & Ρυθμός εκμάθησης:") 
            x0 = float(input("Το x0 είναι το αρχικό σημείο. Παρακαλώ εισάγετε το x0: ")) 
            y0 = float(input("Το y0 είναι το αρχικό σημείο. Παρακαλώ εισάγετε το y0: ")) 
            return x0, y0 

        except ValueError: 
            print("Λάθος είσοδος! Παρακαλώ εισάγετε έγκυρους αριθμούς για x0 και y0.")  

def Parameters(): 
    while True: 
        try: 
            a = float(input("Το a είναι το learning rate. Παρακαλώ εισάγετε το a: ")) 
            print("-------------------------------------------------------------------------------------")

            print("Ο αλγόριθμος της Steepest Descent σταματά αν ένα από τα παρακάτω κριτήρια τερματισμού ικανοποιηθεί: \n"
            "1. Σε περίπτωση που ο αριθμός των επαναλήψεων ξεπεράσει τις 1000 και δεν έχει ικανοποιηθεί κάποιο από τα κριτήρια.\n"
            "2. Η κλίση της συνάρτησης στο σημείο που έχει βρεθεί είναι μικρότερη από μια σταθερά c1 την οποία ορίζει ο χρήστης. \n"
            " Αν η κλίση ισούται με 0 στο αρχικό σημείο (x0,y0), τότε ζητά από τον χρήστη να δώσει νέο αρχικό σημείο.\n"
            " 3. Η απόσταση μεταξύ δύο διαδοχικών σημείων να είναι μικρότερη από μια σταθερά c2 την οποία ορίζει ο χρήστης. \n"
            " 4. Η τιμή της συνάρτησης μεταξύ δύο διαδοχικών σημείων να είναι μικρότερη από μια σταθερά c3 την οποία ορίζει ο χρήστης. \n")
            print("-------------------------------------------------------------------------------------")

            print("Επομένως, για να προχωρήσει ο αλγόριθμός απαιτούνται τα c1, c2, c3 από τον χρήστη:")
            c1 = float(input("1ο) Ποιο το c1 ώστε να μπορεί να εξεταστεί το κριτήριο |∇f| ≤ c1;")) 
            c2 = float(input("2ο) Ποιο το c2 ώστε να μπορεί να εξεταστεί το κριτήριο |Xn - Xn-1| ≤ c2; ")) 
            c3 = float(input("3ο) Ποιο το c3 ώστε να μπορεί να εξεταστεί το κριτήριο |f(Xn) - f(Xn-1)| ≤ c3; ")) 
            return a, c1, c2, c3 

        except ValueError: 
            print("Λάθος είσοδος! Παρακαλώ εισάγετε έγκυρους αριθμούς.") 

def f():
    while True: 
        try: 
            print("-------------------------------------------------------------------------------------")
            expr_str = input("Παρακαλώ εισάγετε τη συνάρτηση σε μορφή x και y (π.χ., x**2 + y**4): ") 
            expr = sp.sympify(expr_str) 
            return expr 

        except sp.SympifyError: 
            print("Λάθος είσοδος! Παρακαλώ εισάγετε έγκυρη συνάρτηση.") 

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
    print("Η μερική παράγωγος ως προς x:", slope_x_expr) 
    print("Η μερική παράγωγος ως προς y:", slope_y_expr) 

    while grad_norm == 0:
        print("Η κλίση στο σημείο (x0, y0) είναι 0. Παρακαλώ εισάγετε νέο σημείο εκκίνησης.")
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
            criterion = "1ο κριτήριο: Η κλίση είναι μικρή." 
            break 

        elif len(points) > 1: 
            prev_x, prev_y = points[-2] 
            distance = sp.sqrt((x0 - prev_x)**2 + (y0 - prev_y)**2).evalf() 

            if distance < c2: 
                criterion = "2ο κριτήριο: Η απόσταση μεταξύ δύο διαδοχικών σημείων είναι μικρή."  
                break 

            f_prev = f_num(prev_x, prev_y) 
            f_current = f_num(x0, y0) 

            if abs(f_current - f_prev) < c3: 
                criterion = "3ο κριτήριο: Η σύγκλιση της συνάρτησης είναι μικρή." 
                break 
                
        x0 = x0 - a * slope_x
        y0 = y0 - a * slope_y

        x_path.append(x0) 
        y_path.append(y0) 
        z_path.append(f_num(x0, y0)) 

        tries += 1

    if tries > MAX_TRIES: 
        criterion = f"Η αναζήτηση απέτυχε: μέγιστες επαναλήψεις ({MAX_TRIES})." 
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
        print("Ο μέγιστος αριθμός επαναλήψεων ξεπεράστηκε.") 
        return
        
    print("-------------------------------------------------------------------------------------")
    print(f"Ελάχιστο σημείο: ({min_x}, {min_y}), με τιμή συνάρτησης f(x, y) = {min_value}") 
    print("Κριτήριο σύγκλισης ->", criterion) 
    print(f"Αριθμός επαναλήψεων: {total_tries}") 

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
