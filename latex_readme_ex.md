$$ \frac{u_{i-1,j} - 2u_{i,j} + u_{i+1,j}}{h^2} + \frac{u_{i,j-1} - 2u_{i,j} + u_{i,j+1}}{h^2} = f_{i,j} $$

$$ \begin{cases} u(0,y) = 0 & u(x,0) = 0 \\ u(1,y) = 0 & u(x,1) = 0 \end{cases} \quad \Rightarrow \quad u(x,y) = \sin(\pi m x) \cdot \sin(\pi l y) $$

$$ \sin(\pi l y_j) \cdot \left( \sin(\pi m(x_i-h)) - 2\sin(\pi m x_i) + \sin(\pi m(x_i+h)) \right) + $$
$$ + \sin(\pi m x_i) \cdot \left( \sin(\pi l(y_j-h)) - 2\sin(\pi l y_j) + \sin(\pi l(y_j+h)) \right) = $$

$$ = \sin(\pi m x_i) \cdot \sin(\pi l y_j) \cdot 2(\cos(\pi m h) - 1) + $$
$$ + \sin(\pi m x_i) \cdot \sin(\pi l y_j) \cdot 2(\cos(\pi l h) - 1) $$

$$ \lambda_{m,l} = 2\cos(\pi m h) + 2\cos(\pi l h) - 4 $$

$$ A \to \alpha = E - D^{-1}A $$
$$ D = \text{diag}(-4), \quad D^{-1} = \text{diag}(-1/4) $$

$$ \mu_{m,l} = 1 - D^{-1}\lambda_{m,l} = 1 + \frac{1}{4}(2\cos(\pi m h) + 2\cos(\pi l h) - 4) $$
$$ \mu_{m,l} = \frac{1}{2}(\cos(\pi m h) + \cos(\pi l h)) $$

$$ h = \frac{1}{n}, \quad m,l = 1, \dots, n-1 $$

1.
$$ \rho(\alpha) = \max_{m,l} |\mu_{m,l}| = \frac{1}{2}\left(\cos\frac{\pi}{n} + \cos\frac{\pi}{n}\right) = \cos\left(\frac{\pi}{n}\right) $$
$$ \rho(\alpha) < 1 \quad \Rightarrow \quad \text{Метод Якоби сходится всегда} $$

2.
$$ \rho(\alpha) \to 1 \quad \text{при} \quad n \to \infty \quad \Rightarrow \quad \text{сходится медленно} $$