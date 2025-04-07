import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from tkinter import *
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RootCalculator:
    def __init__(self, master):
        self.master = master
        master.title("✦ CALCULADORA DE RAÍCES ✦")
        master.geometry("1000x800")
        master.configure(bg='black')
        
        # Configuración de estilo
        self.neon_blue = '#00ffff'
        self.bg_color = 'black'
        self.font_style = ('Courier New', 10, 'bold')
        
        # Frame principal
        self.main_frame = Frame(master, bg=self.bg_color)
        self.main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Variable para la ventana de gráficas
        self.graph_window = None
        
        # Mostrar menú principal
        self.show_main_menu()

    def show_main_menu(self):
        """Muestra el menú principal con los tres métodos"""
        # Limpiar frame si ya existe
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Frame interno para centrado
        inner_frame = Frame(self.main_frame, bg=self.bg_color)
        inner_frame.pack(expand=True)
        
        # Título
        Label(inner_frame, 
              text="✦ MÉTODOS NUMÉRICOS ✦", 
              font=('Courier New', 16, 'bold'),
              fg=self.neon_blue,
              bg=self.bg_color).pack(pady=(0, 30))
        
        # Subtítulo
        Label(inner_frame, 
              text="Seleccione el método:", 
              font=('Courier New', 12),
              fg=self.neon_blue,
              bg=self.bg_color).pack(pady=(0, 20))
        
        # Frame para botones
        btn_frame = Frame(inner_frame, bg=self.bg_color)
        btn_frame.pack(pady=10)
        
        # Botones para cada método
        Button(btn_frame, 
              text="Método de Newton-Raphson", 
              command=self.setup_newton,
              width=25, 
              height=2,
              font=self.font_style,
              bg=self.bg_color,
              fg=self.neon_blue,
              relief=RAISED,
              borderwidth=2).grid(row=0, column=0, padx=10, pady=5)
        
        Button(btn_frame, 
              text="Método de la Secante", 
              command=self.setup_secante,
              width=25, 
              height=2,
              font=self.font_style,
              bg=self.bg_color,
              fg=self.neon_blue,
              relief=RAISED,
              borderwidth=2).grid(row=1, column=0, padx=10, pady=5)
        
        Button(btn_frame, 
              text="Método de Bisección", 
              command=self.setup_biseccion,
              width=25, 
              height=2,
              font=self.font_style,
              bg=self.bg_color,
              fg=self.neon_blue,
              relief=RAISED,
              borderwidth=2).grid(row=2, column=0, padx=10, pady=5)

    def setup_newton(self):
        """Configura la interfaz para el método de Newton-Raphson"""
        self.current_method = "newton"
        self.setup_common_interface("NEWTON-RAPHSON")
        
        # Campo específico para Newton
        Label(self.param_frame, 
              text="APROXIMACIÓN INICIAL (x₀):", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=1, column=0, sticky=W, pady=5)
        
        self.x0_entry = Entry(self.param_frame, 
                            font=self.font_style,
                            bg='#111111',
                            fg=self.neon_blue,
                            insertbackground=self.neon_blue)
        self.x0_entry.grid(row=1, column=1, pady=5, padx=10, sticky=W)
        self.x0_entry.insert(0, "2.0")

    def setup_secante(self):
        """Configura la interfaz para el método de la Secante"""
        self.current_method = "secante"
        self.setup_common_interface("SECANTE")
        
        # Campos específicos para Secante
        Label(self.param_frame, 
              text="PRIMERA APROXIMACIÓN (x₀):", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=1, column=0, sticky=W, pady=5)
        
        self.x0_entry = Entry(self.param_frame, 
                            font=self.font_style,
                            bg='#111111',
                            fg=self.neon_blue,
                            insertbackground=self.neon_blue)
        self.x0_entry.grid(row=1, column=1, pady=5, padx=10, sticky=W)
        self.x0_entry.insert(0, "1.0")
        
        Label(self.param_frame, 
              text="SEGUNDA APROXIMACIÓN (x₁):", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=2, column=0, sticky=W, pady=5)
        
        self.x1_entry = Entry(self.param_frame, 
                            font=self.font_style,
                            bg='#111111',
                            fg=self.neon_blue,
                            insertbackground=self.neon_blue)
        self.x1_entry.grid(row=2, column=1, pady=5, padx=10, sticky=W)
        self.x1_entry.insert(0, "2.0")

    def setup_biseccion(self):
        """Configura la interfaz para el método de Bisección"""
        self.current_method = "biseccion"
        self.setup_common_interface("BISECCIÓN")
        
        # Campos específicos para Bisección
        Label(self.param_frame, 
              text="EXTREMO IZQUIERDO (a):", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=1, column=0, sticky=W, pady=5)
        
        self.a_entry = Entry(self.param_frame, 
                           font=self.font_style,
                           bg='#111111',
                           fg=self.neon_blue,
                           insertbackground=self.neon_blue)
        self.a_entry.grid(row=1, column=1, pady=5, padx=10, sticky=W)
        self.a_entry.insert(0, "1.0")
        
        Label(self.param_frame, 
              text="EXTREMO DERECHO (b):", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=2, column=0, sticky=W, pady=5)
        
        self.b_entry = Entry(self.param_frame, 
                           font=self.font_style,
                           bg='#111111',
                           fg=self.neon_blue,
                           insertbackground=self.neon_blue)
        self.b_entry.grid(row=2, column=1, pady=5, padx=10, sticky=W)
        self.b_entry.insert(0, "2.0")

    def setup_common_interface(self, method_name):
        """Configura la interfaz común para todos los métodos"""
        # Limpiar frame principal
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Botón de regreso
        back_btn = Button(self.main_frame, 
                         text="← MENÚ PRINCIPAL", 
                         command=self.show_main_menu,
                         font=self.font_style,
                         bg=self.bg_color,
                         fg=self.neon_blue,
                         borderwidth=0)
        back_btn.pack(anchor=NW, pady=(0, 20))
        
        # Título del método
        title = Label(self.main_frame, 
                     text=f"✦ MÉTODO DE {method_name} ✦", 
                     font=('Courier New', 14, 'bold'),
                     fg=self.neon_blue,
                     bg=self.bg_color)
        title.pack(pady=(0, 20))
        
        # Frame para parámetros
        self.param_frame = Frame(self.main_frame, bg=self.bg_color)
        self.param_frame.pack(fill=X, padx=50)
        
        # Función a evaluar
        Label(self.param_frame, 
              text="FUNCIÓN f(x):", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=0, column=0, sticky=W, pady=5)
        
        self.func_entry = Entry(self.param_frame, 
                              font=self.font_style,
                              bg='#111111',
                              fg=self.neon_blue,
                              insertbackground=self.neon_blue,
                              width=30)
        self.func_entry.grid(row=0, column=1, pady=5, padx=10)
        self.func_entry.insert(0, "x**3 - 2*x - 5")
        
        # Tolerancia
        Label(self.param_frame, 
              text="TOLERANCIA:", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=3, column=0, sticky=W, pady=5)
        
        self.tol_entry = Entry(self.param_frame, 
                             font=self.font_style,
                             bg='#111111',
                             fg=self.neon_blue,
                             insertbackground=self.neon_blue)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.grid(row=3, column=1, pady=5, padx=10, sticky=W)
        
        # Iteraciones máximas
        Label(self.param_frame, 
              text="ITERACIONES MÁXIMAS:", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).grid(row=4, column=0, sticky=W, pady=5)
        
        self.max_iter_entry = Entry(self.param_frame, 
                                  font=self.font_style,
                                  bg='#111111',
                                  fg=self.neon_blue,
                                  insertbackground=self.neon_blue)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=4, column=1, pady=5, padx=10, sticky=W)
        
        # Botón de cálculo
        calc_btn = Button(self.main_frame, 
                         text="CALCULAR RAÍZ ✦", 
                         command=self.calculate_root,
                         font=('Courier New', 12, 'bold'),
                         bg=self.bg_color,
                         fg=self.neon_blue,
                         activebackground=self.bg_color,
                         activeforeground=self.neon_blue,
                         relief=RAISED,
                         borderwidth=4)
        calc_btn.pack(pady=20)
        
        # Frame para resultados
        self.result_frame = Frame(self.main_frame, bg=self.bg_color)
        self.result_frame.pack(fill=BOTH, expand=True)

    def calculate_root(self):
        """Calcula la raíz según el método seleccionado"""
        try:
            # Limpiar resultados anteriores
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            # Obtener parámetros comunes
            f_str = self.func_entry.get()
            tol = float(self.tol_entry.get())
            max_iter = int(self.max_iter_entry.get())
            
            # Crear función lambda
            f = lambda x: eval(f_str, {'x': x, 'math': math, 'np': np})
            
            # Cerrar ventana de gráficas si está abierta
            if self.graph_window:
                self.graph_window.destroy()
            
            # Aplicar método seleccionado
            if self.current_method == "newton":
                results, root, iterations = self.newton_raphson(f, tol, max_iter)
            elif self.current_method == "secante":
                results, root, iterations = self.secante(f, tol, max_iter)
            else:  # bisección
                results, root, iterations = self.biseccion(f, tol, max_iter)
            
            # Mostrar resultados
            self.show_results(root, iterations)
            self.show_iterations_table(results)
            self.show_convergence_graphs(results)
            
        except Exception as e:
            messagebox.showerror("ERROR", f"Datos inválidos: {str(e)}")

    def newton_raphson(self, f, tol, max_iter):
        """
        Método de Newton-Raphson para encontrar raíces
        xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
        """
        # Derivación simbólica
        x = sp.symbols('x')
        expr = sp.sympify(self.func_entry.get())
        df_expr = sp.diff(expr, x)
        df = sp.lambdify(x, df_expr, modules=['numpy', 'math'])
        
        # Mostrar derivada
        self.show_derivative(df_expr)
        
        # Obtener aproximación inicial
        x0 = float(self.x0_entry.get())
        
        # Almacenar resultados
        results = []
        iterations = 0
        error = float('inf')
        
        while iterations < max_iter and error > tol:
            fx = f(x0)
            dfx = df(x0)
            
            if abs(dfx) < 1e-15:
                raise ValueError("Derivada cero en Newton-Raphson")
            
            x_new = x0 - fx / dfx
            error = abs(x_new - x0)
            
            results.append({
                'iteracion': iterations + 1,
                'x_n': x0,
                'f(x_n)': fx,
                'f\'(x_n)': dfx,
                'x_n+1': x_new,
                'error': error
            })
            
            x0 = x_new
            iterations += 1
        
        return results, x0, iterations

    def secante(self, f, tol, max_iter):
        """
        Método de la Secante para encontrar raíces
        xₙ₊₁ = xₙ - f(xₙ)[(xₙ - xₙ₋₁)/(f(xₙ) - f(xₙ₋₁))]
        """
        # Obtener aproximaciones iniciales
        x0 = float(self.x0_entry.get())
        x1 = float(self.x1_entry.get())
        
        # Verificar cambio de signo
        if f(x0) * f(x1) >= 0:
            messagebox.showwarning("ADVERTENCIA", 
                                 "f(x₀) y f(x₁) deben tener signos opuestos para garantizar convergencia")
            return [], None, 0
        
        # Almacenar resultados
        results = []
        iterations = 0
        error = float('inf')
        
        while iterations < max_iter and error > tol:
            fx0 = f(x0)
            fx1 = f(x1)
            
            denominator = fx1 - fx0
            if abs(denominator) < 1e-15:
                raise ValueError("División por cero en el método de la Secante")
            
            x_new = x1 - fx1 * (x1 - x0) / denominator
            error = abs(x_new - x1)
            
            results.append({
                'iteracion': iterations + 1,
                'x_n-1': x0,
                'x_n': x1,
                'f(x_n-1)': fx0,
                'f(x_n)': fx1,
                'x_n+1': x_new,
                'error': error
            })
            
            x0, x1 = x1, x_new
            iterations += 1
        
        return results, x1, iterations

    def biseccion(self, f, tol, max_iter):
        """
        Método de Bisección para encontrar raíces
        Requiere que f(a) y f(b) tengan signos opuestos
        """
        # Obtener intervalo inicial
        a = float(self.a_entry.get())
        b = float(self.b_entry.get())
        
        # Verificar cambio de signo
        if f(a) * f(b) >= 0:
            messagebox.showwarning("ADVERTENCIA", 
                                 "f(a) y f(b) deben tener signos opuestos para aplicar el método")
            return [], None, 0
        
        # Almacenar resultados
        results = []
        iterations = 0
        error = (b - a) / 2
        
        while iterations < max_iter and error > tol:
            c = (a + b) / 2
            fc = f(c)
            error = (b - a) / 2
            
            results.append({
                'iteracion': iterations + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(a)': f(a),
                'f(b)': f(b),
                'f(c)': fc,
                'error': error
            })
            
            if fc == 0:
                break
            elif f(a) * fc < 0:
                b = c
            else:
                a = c
            
            iterations += 1
        
        root = (a + b) / 2
        return results, root, iterations

    def show_derivative(self, df_expr):
        """Muestra la derivada calculada para Newton-Raphson"""
        derivative_frame = Frame(self.result_frame, bg=self.bg_color)
        derivative_frame.pack(pady=(0, 10), fill=X)
        
        Label(derivative_frame, 
              text="Derivada calculada:", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).pack(side=LEFT)
        
        Label(derivative_frame, 
              text=f"f'(x) = {sp.pretty(df_expr)}", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).pack(side=LEFT, padx=10)

    def show_results(self, root, iterations):
        """Muestra los resultados finales del cálculo"""
        result_frame = Frame(self.result_frame, bg=self.bg_color)
        result_frame.pack(pady=(10, 20), fill=X)
        
        Label(result_frame, 
              text=f"✦ RESULTADOS ✦", 
              font=('Courier New', 12, 'bold'), 
              fg=self.neon_blue,
              bg=self.bg_color).pack()
        
        Label(result_frame, 
              text=f"Raíz aproximada: {root:.12f}", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).pack()
        
        Label(result_frame, 
              text=f"Iteraciones realizadas: {iterations}", 
              font=self.font_style,
              fg=self.neon_blue,
              bg=self.bg_color).pack()

    def show_iterations_table(self, results):
        """Muestra una tabla con todas las iteraciones"""
        if not results:
            return
            
        # Frame para la tabla
        table_frame = Frame(self.result_frame, bg=self.bg_color)
        table_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar
        scroll_y = Scrollbar(table_frame)
        scroll_y.pack(side=RIGHT, fill=Y)
        
        # Configurar columnas según el método
        if self.current_method == "newton":
            columns = ('iteracion', 'x_n', 'f(x_n)', 'f\'(x_n)', 'x_n+1', 'error')
            headings = ['Iteración', 'xₙ', 'f(xₙ)', 'f\'(xₙ)', 'xₙ₊₁', 'Error |xₙ₊₁-xₙ|']
            col_widths = [80, 120, 120, 120, 120, 120]
        elif self.current_method == "secante":
            columns = ('iteracion', 'x_n-1', 'x_n', 'f(x_n-1)', 'f(x_n)', 'x_n+1', 'error')
            headings = ['Iteración', 'xₙ₋₁', 'xₙ', 'f(xₙ₋₁)', 'f(xₙ)', 'xₙ₊₁', 'Error |xₙ₊₁-xₙ|']
            col_widths = [80, 100, 100, 100, 100, 100, 120]
        else:  # bisección
            columns = ('iteracion', 'a', 'b', 'c', 'f(a)', 'f(b)', 'f(c)', 'error')
            headings = ['Iteración', 'a', 'b', 'c', 'f(a)', 'f(b)', 'f(c)', 'Error (b-a)/2']
            col_widths = [80, 100, 100, 100, 100, 100, 100, 120]
        
        # Crear tabla
        tree = ttk.Treeview(table_frame, columns=columns, yscrollcommand=scroll_y.set, height=8)
        
        # Configurar columnas
        tree.heading('#0', text='#')
        for col, heading, width in zip(columns, headings, col_widths):
            tree.heading(col, text=heading)
            tree.column(col, width=width, anchor=CENTER)
        
        tree.column('#0', width=0, stretch=NO)
        
        # Insertar datos
        for row in results:
            values = [row[col] for col in columns]
            formatted_values = [f"{val:.8f}" if isinstance(val, (float, int)) else str(val) 
                              for val in values]
            tree.insert('', 'end', values=formatted_values)
        
        tree.pack(fill=BOTH, expand=True)
        scroll_y.config(command=tree.yview)
        
        # Estilo de la tabla
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", 
                      background="#111111", 
                      foreground=self.neon_blue,
                      fieldbackground="#111111",
                      font=('Courier New', 9))
        style.configure("Treeview.Heading", 
                       background="#003333", 
                       foreground=self.neon_blue,
                       font=('Courier New', 9, 'bold'))
        style.map('Treeview', background=[('selected', '#006666')])

    def show_convergence_graphs(self, results):
        """Muestra las gráficas de convergencia en una ventana separada"""
        if not results:
            return
            
        # Crear ventana para gráficas
        self.graph_window = Toplevel(self.master)
        self.graph_window.title(f"Convergencia - {self.current_method.upper()}")
        self.graph_window.geometry("900x700")
        self.graph_window.configure(bg='black')
        
        # Frame para gráficas
        graph_frame = Frame(self.graph_window, bg='black')
        graph_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Crear figura de matplotlib
        figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        figure.set_facecolor('black')
        
        # Configurar estilo de los ejes
        for ax in [ax1, ax2]:
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color(self.neon_blue)
            ax.tick_params(axis='x', colors=self.neon_blue)
            ax.tick_params(axis='y', colors=self.neon_blue)
            ax.title.set_color(self.neon_blue)
            ax.grid(True, color='#222222')
        
        # Datos comunes
        iterations = [r['iteracion'] for r in results]
        errors = [r['error'] for r in results]
        
        # Gráfico 1: Convergencia del valor
        if self.current_method == "newton":
            x_values = [r['x_n+1'] for r in results]
            ax1.plot(iterations, x_values, 'o-', color=self.neon_blue, label='xₙ')
            ax1.set_title('Convergencia del valor (Newton-Raphson)')
        elif self.current_method == "secante":
            x_values = [r['x_n+1'] for r in results]
            ax1.plot(iterations, x_values, 'o-', color=self.neon_blue, label='xₙ')
            ax1.set_title('Convergencia del valor (Secante)')
        else:  # bisección
            a_values = [r['a'] for r in results]
            b_values = [r['b'] for r in results]
            c_values = [r['c'] for r in results]
            ax1.plot(iterations, a_values, 'o-', color='#00ff00', label='a')
            ax1.plot(iterations, b_values, 'o-', color='#ff00ff', label='b')
            ax1.plot(iterations, c_values, 'o-', color=self.neon_blue, label='c')
            ax1.set_title('Convergencia del intervalo (Bisección)')
        
        ax1.set_xlabel('Iteración', color=self.neon_blue)
        ax1.legend(facecolor='#111111', edgecolor=self.neon_blue)
        
        # Gráfico 2: Convergencia del error (escala logarítmica)
        ax2.semilogy(iterations, errors, 'o-', color=self.neon_blue)
        ax2.set_title('Convergencia del error (escala logarítmica)')
        ax2.set_xlabel('Iteración', color=self.neon_blue)
        ax2.set_ylabel('Error', color=self.neon_blue)
        
        # Ajustar layout
        figure.tight_layout()
        
        # Mostrar en tkinter
        canvas = FigureCanvasTkAgg(figure, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# Ejecutar la aplicación
if __name__ == "__main__":
    root = Tk()
    app = RootCalculator(root)
    root.mainloop()