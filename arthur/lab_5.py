import streamlit as st
import numpy as np
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def U(x, t, a):
    return np.exp(-4 * np.pi * a * t) * np.sin(2 * np.pi * x)

def ui_0(x):
    return np.sin(2 * np.pi * x)

def mu_1(t):
    return 0

def mu_2(t):
    return 0

def real_solve(tau, layers_count, h, a):
    x_count = int(1 / h)  + 1
    layers = np.zeros((layers_count, x_count))

    for i in range(layers_count):
        layers[i] = np.array([U(j * h, i * tau, a) for j in range(x_count)])

    return layers

def explicit_solve(tau, layers_count, h, a):
    x_count = int(1 / h) + 1
    layers = np.zeros((layers_count, x_count))

    first_layer = [ui_0(j * h) for j in range(x_count)]
    layers[0] = np.array(first_layer)
    for i in range(1, layers_count):
        layer = [mu_1(i * tau)] + \
            [layers[i-1][j] + 
            a * tau * (layers[i-1][j+1] - 2 * layers[i-1][j] + 
            layers[i-1][j-1]) / (h**2) 
            for j in range(1, x_count - 1)] + \
            [mu_2(i * tau)]
        layers[i] = np.array(layer)

    return layers

def implicit_solve(tau, layers_count, h, a):
    x_count = int(1 / h) + 1
    layers = np.zeros((layers_count, x_count))
    alpha = np.zeros(x_count - 1)
    beta = np.zeros(x_count - 1)

    sigma = a * tau / (h**2)
    aa = -sigma
    bb = 1 + 2 * sigma
    cc = -sigma

    if abs(bb) < abs(aa) + abs(cc):
        st.markdown('### Метод прогонки может не сходиться!')

    first_layer = [ui_0(j * h) for j in range(x_count)]
    layers[0] = np.array(first_layer)
    for i in range(1, layers_count):
        layer = np.zeros(x_count)
        beta[0] = mu_1(i * tau)
        for j in range(1, x_count - 1):
            alpha[j] = -aa / (bb + cc * alpha[j - 1])
            beta[j] = (layers[i-1][j] - cc * beta[j-1]) / (bb + cc * alpha[j-1])
        layer[x_count - 1] = mu_2(i * tau)
        for j in range(x_count - 2, -1, -1):
            layer[j] = layer[j+1] * alpha[j] + beta[j]
        layers[i] = np.array(layer)

    return layers

def CN_solve(tau, layers_count, h, a):
    x_count = int(1 / h) + 1
    layers = np.zeros((layers_count, x_count))
    alpha = np.zeros(x_count - 1)
    beta = np.zeros(x_count - 1)

    sigma = a * tau / h**2
    aa = -sigma / 2
    bb = 1 + sigma
    cc = -sigma / 2

    if abs(bb) < abs(aa) + abs(cc):
        st.markdown('### Метод прогонки может не сходиться!')

    first_layer = [ui_0(j * h) for j in range(x_count)]
    layers[0] = np.array(first_layer)
    for i in range(1, layers_count):
        layer = np.zeros(x_count)
        beta[0] = mu_1(i * tau)
        for j in range(1, x_count - 1):
            alpha[j] = -aa / (bb + cc * alpha[j - 1])
            beta[j] = (layers[i-1][j] - \
                aa * (layers[i-1][j+1] - 2 * layers[i-1][j] + layers[i-1][j-1]) - \
                cc * beta[j-1]) / (bb + cc * alpha[j-1])
        layer[x_count - 1] = mu_2(i * tau)
        for j in range(x_count - 2, -1, -1):
            layer[j] = layer[j+1] * alpha[j] + beta[j]
        layers[i] = np.array(layer)

    return layers

def MSE(y, y_hat):
    mse = np.sum((y - y_hat)**2) / len(y)
    return mse

def compare_results(scheme_res, real_res, x, t, **params):
    a, h, tau, layers_count = params.values()

    fig = plt.figure(figsize=(15, 50))
    fst = fig.add_subplot(6, 1, 1)
    scd = fig.add_subplot(6, 1, 2)
    thd = fig.add_subplot(6, 1, 3)
    frth = fig.add_subplot(6, 1, 4, projection='3d')
    fvth = fig.add_subplot(6, 1, 5, projection='3d')
    sxth = fig.add_subplot(6, 1, 6, projection='3d')

    fst.set_title('Аналитическое решение. Результаты на всей сетке', fontsize=20)
    fst.set_xlabel('x', fontsize=20)
    fst.set_ylabel('U(x, t)', fontsize=20)

    for i in range(len(real_res)):
        #fst.scatter(x, layer, marker='x')
        if i % 4 == 0:
            fst.plot(x, real_res[i])
    fst.legend(['t = {}'.format(i * tau) for i in range(layers_count)], fontsize=20, loc='upper right')

    scd.set_title('Cхема. Результаты на всей сетке', fontsize=20)
    scd.set_xlabel('x', fontsize=20)
    scd.set_ylabel('U(x, t)', fontsize=20)

    for i in range(len(scheme_res)):
        #scd.scatter(x, layer, marker='x')
        if i % 4 == 0:
            scd.plot(x, scheme_res[i])
    scd.legend(['t = {}'.format(i * tau) for i in range(layers_count)], fontsize=20, loc='upper right')

    MSE_error = np.array([MSE(i, j) for i, j in zip(scheme_res, real_res)])
    thd.set_title('MSE: h = {}, tau = {}'.format(h, tau), fontsize=20)
    thd.set_xlabel('t', fontsize=20)
    thd.set_ylabel('mse_error', fontsize=20)
    thd.plot(t, MSE_error)
    thd.scatter(t, MSE_error, marker='o', c='r', s=50)
    thd.legend(['Значение ошибки в разные моменты времени'], fontsize=20, loc='upper right')

    X, T = np.meshgrid(x, t)
    U_real = real_res.ravel().reshape(X.shape)
    frth.set_title('Аналитическое решение. Результаты на всей сетке, 3D', fontsize=20)
    frth.set_xlabel('x', fontsize=20)
    frth.set_ylabel('t', fontsize=20)
    frth.set_zlabel('u', fontsize=20)
    surf_real = frth.plot_surface(X, T, U_real, rstride=1, cstride=1, cmap=cm['coolwarm'],
                           linewidth=0, antialiased=False)
    fig.colorbar(surf_real, ax=frth, shrink=0.5, aspect=10)

    U_exp = scheme_res.ravel().reshape(X.shape)
    fvth.set_title('Cхема. Результаты на всей сетке, 3D', fontsize=20)
    fvth.set_xlabel('x', fontsize=20)
    fvth.set_ylabel('t', fontsize=20)
    fvth.set_zlabel('u', fontsize=20)
    surf_explicit = fvth.plot_surface(X, T, U_exp, rstride=1, cstride=1, cmap=cm['coolwarm'],
                           linewidth=0, antialiased=False)
    fig.colorbar(surf_explicit, ax=fvth, shrink=0.5, aspect=10)

    U_real = real_res.ravel().reshape(X.shape)
    U_exp = scheme_res.ravel().reshape(X.shape)
    U_err = (U_real - U_exp)**2

    sxth.set_title('Ошибка. Результаты на всей сетке, 3D', fontsize=20)
    sxth.set_xlabel('x', fontsize=20)
    sxth.set_ylabel('t', fontsize=20)
    sxth.set_zlabel('u', fontsize=20)
    surf_err= sxth.plot_surface(X, T, U_err, rstride=1, cstride=1, cmap=cm['hsv'],
                           linewidth=0, antialiased=False)
    fig.colorbar(surf_err, ax=sxth, shrink=0.5, aspect=10)

    st.pyplot(fig)

def difference_scheme():
    params = {
        'a': 0.01,
        'h':  0.1,
        'tau': 0.1,
        'layers_count': 5
    }
    a, h, tau, layers_count = params.values()
    with st.sidebar:
        a = float(st.text_input('a = ', value=str(a), key=1))
        h = float(st.text_input('h = ', value=str(h), key=2))
        tau = float(st.text_input('tau = ', value=str(tau), key=3))
        layers_count = st.number_input('layers count = ', min_value=1, max_value=500, value=layers_count)

    st.write(f'Параметр a = {a}')
    st.write(f'Шаг по пространственной шкале h = {h}')
    st.write(f'Шаг по временной шкале tau = {tau}')
    st.write(f'Число слоев сетки = {layers_count}')

    # Вычисляем здесь, чтобы обновлялось вместе с изменением параметров
    x_count = int(1 / h) + 1
    x = np.array([i * h for i in range(x_count)])
    t = np.array([i * tau for i in range(layers_count)])

    real_res = real_solve(tau, layers_count, h, a)
    graph_type = None

    with st.sidebar:
        if st.button('Явная схема'):
            graph_type = 'expllicit'
        if st.button('Неявная схема'):
            graph_type = 'impllicit'
        if st.button('Cхема Кранка-Николсона'):
            graph_type = 'cn'

    if graph_type == 'expllicit':
        st.markdown('### Явная схема')
        if a * tau / h**2 > 0.5:
            st.markdown('#### Предупреждение: схема может быть неустойчивой!')
        st.markdown('###### Конечно-разностная схема')
        st.latex(r'''
    \begin{gather*}
    \frac{u_{i}^{n+1} - u_{i}^{n}}{\tau} = 
    a \frac{u_{i+1}^{n} - 2u_{i}^{n} + u_{i-1}^{n}}{h^{2}}, \\ i = 1,\dots,N-1, \ \ n=0,\dots,K-1, \ \ hN = 1, \ \ \tau K  =T 
    \end{gather*}
    ''')
        st.markdown('Начальные условия:')
        st.latex(r'''
    u_{0}^{n} = \mu_{1}(t_{n}), \ \  u_{N}^{n} = \mu_{2}(t_{n}), \ \ n = 0,1, \dots, K
    ''')
        st.markdown('Граничные условия:')
        st.latex(r'''
    u_{i}^{0} = u_{0}(x_{i}), \ \ i = 0, 1, \dots, N
    ''')
        st.markdown('Решение:')
        st.latex(r'''
        \displaystyle \sigma = \frac{a \tau}{h^{2}}
        ''')
        st.latex(r'''
    u_{i}^{n+1} = u_{i}^{n} + \sigma (u_{i-1}^{n} - 2 u_{i}^{n} + u_{i+1}^{n})
        ''')
        st.latex(r'''
    u_{i}^{n+1} = \sigma u_{i-1}^{n} + (1 - 2 \sigma) u_{i}^{n} + \sigma u_{i+1}^{n}
        ''')
        explicit_res = explicit_solve(tau, layers_count, h, a)
        compare_results(explicit_res, real_res, x, t, **params)
    if graph_type == 'impllicit':
        st.markdown('### Невная схема')
        st.markdown('###### Конечно-разностная схема')
        st.latex(r'''
    \begin{gather*}
    \frac{u_{i}^{n+1} - u_{i}^{n}}{\tau} = 
    a \frac{u_{i+1}^{n+1} - 2u_{i}^{n+1} + u_{i-1}^{n+1}}{h^{2}} \\ 
    i = 1,\dots,N-1, \ \ n=0,\dots,K-1, \ \ hN = 1, \ \ \tau K  = T
    \end{gather*}
    ''')
        st.markdown('Начальные условия:')
        st.latex(r'''
    u_{0}^{n} = \mu_{1}(t_{n}), \ \  u_{N}^{n} = \mu_{2}(t_{n}), \ \ n = 0,1, \dots, K
    ''')
        st.markdown('Граничные условия:')
        st.latex(r'''
    u_{i}^{0} = u_{0}(x_{i}), \ \ i = 0, 1, \dots, N
    ''')
        st.markdown('Решение:')
        st.latex(r'''
        \sigma = \frac{a \tau}{h^{2}}
        ''')
        st.latex(r'''
    u_{i}^{n+1} - u_{i}^{n} = \sigma (u_{i-1}^{n+1} - 2u_{i}^{n+1} + u_{i+1}^{n+1})
        ''')
        st.latex(r'''
    -\sigma u_{i+1}^{n+1} + (1 + 2\sigma)u_{i}^{n+1} - \sigma u_{i+1}^{n+1} = u_{i}^{n}
        ''')
        implicit_res = implicit_solve(tau, layers_count, h, a)
        compare_results(implicit_res, real_res, x, t, **params)
    if graph_type == 'cn':
        st.markdown('### Схема Кранка-Николсона')
        st.markdown('###### Конечно-разностная схема')
        st.latex(r'''
    \begin{gather*}
    \frac{u_{i}^{n+1} - u_{i}^{n}}{\tau} 
= \frac{a}{2} \cdot \left(\frac{u_{i+1}^{n} - 2u_{i}^{n} + u_{i-1}^{n}}{h^{2}} + \frac{u_{i+1}^{n+1} - 2u_{i}^{n+1} + u_{i-1}^{n+1}}{h^{2}} \right) \\
    i = 1,\dots,N-1, \ \ n=0,\dots,K-1, \ \ hN = 1, \ \ \tau K  =T
    \end{gather*}
        ''')
        st.markdown('Начальные условия:')
        st.latex(r'''
    u_{0}^{n} = \mu_{1}(t_{n}), \ \  u_{N}^{n} = \mu_{2}(t_{n}), \ \ n = 0,1, \dots, K
    ''')
        st.markdown('Граничные условия:')
        st.latex(r'''
    u_{i}^{0} = u_{0}(x_{i}), \ \ i = 0, 1, \dots, N
    ''')
        st.markdown('Решение:')
        st.latex(r'''\sigma = \frac{a \tau}{h^2}''')
        st.latex(r'''
    u_{i}^{n+1} - u_{i}^{n}
= \frac{\sigma}{2} \cdot (u_{i-1}^{n} - 2u_{i}^{n} + u_{i+1}^{n} + u_{i-1}^{n+1} - 2u_{i}^{n+1} + u_{i+1}^{n+1})
        ''')
        st.latex(r'''
    -\frac{\sigma}{2} u_{i-1}^{n+1} + (1 + \sigma) u_{i}^{n+1} - \frac{\sigma}{2} u_{i+1}^{n+1} = \frac{\sigma}{2} u_{i-1}^{n} + (1 - \sigma)u_{i}^{n} + \frac{\sigma}{2} u_{i+1}^{n}
        ''')
        cn_res = CN_solve(tau, layers_count, h, a)
        compare_results(cn_res, real_res, x, t, **params)

def main():
    st.title('Лабораторная 5')
    st.markdown('### Вариант 1')
    st.latex(r'''
    \frac{ \partial u }{ \partial t } = 
    a \frac{ \partial^{2} u }{ \partial x^{2} }, \ a > 0 
    ''')
    st.latex(r'''
    u(0, t) = 0, \\
    u(1, t) = 0, \\
    u(x, 0) = \sin(2 \pi x)
    ''')
    st.write('Аналитическое решение:')
    st.latex(r'''
    U(x, t) = \exp(-4 \pi^2 at) \sin(2 \pi x)
    ''')
    st.markdown('## Конечно-разностные схемы')

    difference_scheme()

main()
