# Definir dos vectores (listas): input my_x, pesos my_w
my_x = [0, 1]#input un item
my_w = [0.66, 0.80]

# Test la función mul() con un item my_x 
# y los pesos descubiertos en clase my_w, el resultado debería ser 
# el vector [0.0,0.8]
mul(my_x, my_w)

# Multiplicar dos vectores elemento a elemento
def mul(a, b):
    """
    devolver una lista c, de la misma longitud que a y b donde 
    cada elemento c[i] = a[i] * b[i]
    lo podéis hacer con un bucle o con una list comprenhension
    """
    c = [0] * len(a)
    for i in range(len(a)):
        c [i] = a[i] * b[i]
    return c

# Define el bias my_bias y el peso descubierto en clase asociado a ese bias
# añadiré el bias a el vector de pesos my_w generando un nuevo vector my_wPlusWBias.
# Posibles errores: Recordad que en Python las variables con punteros
# y el insertar si lo ejecutáis varias veces los valores 
# se van acumulando dependiendo de cómo hagáis la inserción
# my_wPlusWBias debería contener [-0.97, 0.66, 0.8]. Pista para hacer copias de un vector. copiaV=v[:] o copiaV=v.copy()

my_bias  = 1
my_wbias = -0.97

my_wPlusWBias = my_w.copy()
my_wPlusWBias.insert(0, my_wbias)
print(my_wPlusWBias)

# Neurona lineal
def distanciaDelCoseno(x, weights, bias):
    """
    El producto escalar (producto punto) de dos vectores y la similitud de coseno no son completamente equivalentes 
    ya que la similitud del coseno solo se preocupa por la diferencia de ángulo, 
    mientras que el producto de punto se preocupa por el ángulo y la magnitud
    Pero en muchas ocasiones se emplean indistintamente
    Así pues, esta función devuelve el valor escalar de la neurona, es decir, 
    el producto escalar entre el vector de entrada añadiendo el bias y el vector de los pesos
    recordad que "sum(list)" computa la suma de los elementos de una lista
    Así pues se comenzará por añadir el bías en la posición 0 del vector de entrada 
    antes de llevar a cabo el producto escalar para así tener dos vectores de 
    la misma longitud. Emplea la función mul que ya has programado
    """
    xPlusBias = x.copy()
    xPlusBias.insert(0, bias)
    mulResult = mul(xPlusBias, weights)
    return sum(mulResult)