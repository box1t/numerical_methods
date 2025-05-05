import math

def f(x):
    return 1 / (256 - x ** 4)

def splitting(x0, xk, h):
    xs = []
    x = x0
    while x < xk:
        xs.append(x)
        x += h
    xs.append(xk)
    return xs


def rectangles(x0, xk, h):
    integral = 0
    xs = splitting(x0, xk, h)
    for i in range(1, len(xs)):
        integral += h * f((xs[i - 1] + xs[i]) / 2)
    return integral


def trapezoids(x0, xk, h):
    integral = 0
    xs = splitting(x0, xk, h)
    for i in range(1, len(xs)):
        integral += 0.5 * h * (f(xs[i - 1]) + f(xs[i]))
    return integral


def simpson(x0, xk, h):
    integral = 0
    xs = splitting(x0, xk, h)
    for i in range(1, len(xs)):
        integral += 1/3 * h/2 * (f(xs[i - 1]) + 4 * f((xs[i - 1] + xs[i]) / 2) + f(xs[i]))
    return integral


def rungeError(values, hs, p):
    k = hs[0] / hs[1]
    return (values[1] - values[0]) / (k ** p - 1)

def runge(values, hs, p):
    return values[1] + rungeError(values, hs, p)


orderRectangles = 2
orderTrapezoids = 2
orderSimpson = 4


x0 = -2
xk = 2
hs = [1, 0.5]
integral = (math.log(6) - math.log(2) + 2 * math.atan(0.5)) / 128

integralsRectangles = []
errorsRectangles = []
integralsTrapezoids = []
errorsTrapezoids = []
integralsSimpson = []
errorsSimpson = []

print(f"Истинное значение: {integral}")

for h in hs:
    print(f"Шаг {h}")
    integralRectangles = rectangles(x0, xk, h)
    integralsRectangles.append(integralRectangles)
    errorRectangles = abs(integral - integralRectangles)
    errorsRectangles.append(errorRectangles)
    print(f"Метод прямоугольников")
    print(f"\tЗначение: {integralRectangles}")
    print(f"\tАбсолютная погрешность: {errorRectangles}")

    integralTrapezoids = trapezoids(x0, xk, h)
    integralsTrapezoids.append(integralTrapezoids)
    errorTrapezoids = abs(integral - integralTrapezoids)
    errorsTrapezoids.append(errorTrapezoids)
    print(f"Метод трапеций")
    print(f"\tЗначение: {integralTrapezoids}")
    print(f"\tАбсолютная погрешность: {errorTrapezoids}")

    integralSimpson = simpson(x0, xk, h)
    integralsSimpson.append(integralSimpson)
    errorSimpson = abs(integral - integralSimpson)
    errorsSimpson.append(errorSimpson)
    print(f"Метод Симпсона")
    print(f"\tЗначение: {integralSimpson}")
    print(f"\tАбсолютная погрешность: {errorSimpson}")

    print("==================================================")

print("Уточненные значения")
integralRectangles = runge(integralsRectangles, hs, orderRectangles)
errorRectangles = abs(integral - integralRectangles)
print(f"Метод прямоугольников")
print(f"\tАпостериорная оценка: {errorsRectangles[1] - errorRectangles}")
print(f"\tЗначение: {integralRectangles}")
print(f"\tАбсолютная погрешность: {errorRectangles}")

integralTrapezoids = runge(integralsTrapezoids, hs, orderTrapezoids)
errorTrapezoids = abs(integral - integralTrapezoids)
print(f"Метод трапеций")
print(f"\tАпостериорная оценка: {errorsTrapezoids[1] - errorTrapezoids}")
print(f"\tЗначение: {integralTrapezoids}")
print(f"\tАбсолютная погрешность: {errorTrapezoids}")

integralSimpson = runge(integralsSimpson, hs, orderSimpson)
errorSimpson = abs(integral - integralSimpson)
print(f"Метод Симпсона")
print(f"\tАпостериорная оценка: {errorsSimpson[1] - errorSimpson}")
print(f"\tЗначение: {integralSimpson}")
print(f"\tАбсолютная погрешность: {errorSimpson}")