import numpy as np
import matplotlib.pyplot as plt

w = 400
h = 300

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

#Parte 1
def intersect_triangle(O, D, V0, V1, V2):

    epsilon = 1e-8

    e1 = V1 - V0
    e2 = V2 - V0

    # Calculamos el determinante
    P = np.cross(D, e2)
    det = np.dot(e1, P)

    # Si el determinante es cercano a cero el rayo está en el plano del triángulo, en otro caso no
    # habrá ninguna intersección.
    if det > -epsilon and det < epsilon:
        return np.inf

    inv_det = 1.0 / det

    # Calculamos la distancia de V0 al rayo origen.
    T = O - V0

    u = np.dot(T, P) * inv_det
    if u < 0.0 or u > 1.0:
        return np.inf

    Q = np.cross(T, e1)

    v = np.dot(D, Q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return np.inf

    t = np.dot(e2, Q) * inv_det

    if t > epsilon:
        return t

    # Si no se cumple ninguna condición no existe la intersección.
    return np.inf


#Parte 2 Modificar la función intersect para que acepte el triángulo.
def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['vertices'][0], obj['vertices'][1], obj['vertices'][2])

#Parte 3 obtener la normal del triángulo
def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        v0, v1, v2 = obj['vertices']
        N = np.cross(v1 - v0, v2 - v0)
        N = normalize(N)
    return N

def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)

    # Start computing the color.
    col_ray = ambient
    for light in lights:
        toL = normalize(light['position'] - M)
        toO = normalize(O - M)
        # Shadow: find if the point is shadowed or not.
        l = [intersect(M + N * .0001, toL, obj_sh)
             for k, obj_sh in enumerate(scene) if k != obj_idx]
        if l and min(l) < np.inf:
            continue
        light_color = light['color']
        # Lambert shading (diffuse).
        col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color * light_color
        # Blinn-Phong shading (specular).
        col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * light_color

    return obj, M, N, col_ray

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color), reflection=.5)

def add_plane(position, normal):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=lambda M: (color_plane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
                diffuse_c=.75, specular_c=.5, reflection=.25)

#Parte 4 Implementar la función de añadir triángulo.
def add_triangle(v0, v1, v2, color):
    return dict(
        type='triangle',
        vertices=[np.array(v0), np.array(v1), np.array(v2)],
        color=np.array(color),
        reflection=0.2
    )

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)

scene = [add_sphere([ 0.25, .1, 2.55], .4, [1., .572, .184]),
         add_sphere([-.75, .1, 2.25], .3, [.5, .223, .5]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
         add_triangle([-1., -0.5, 2.], [0.5, 1., 2.], [1., -0.5, 2.], [1., 1., 0.])
         ]



# Light positions and colors.
lights = [{'position': np.array([5., 5., -10.]), 'color': np.ones(3)},
          {'position': np.array([-3., 5., -15.]), 'color': np.array([0.5, 0.5, 0.5])},
          {'position': np.array([0., 10., -5.]), 'color': np.array([0.2, 0.2, 0.8])}]

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50


depth_max = 5  # Maximum number of light reflections.
col = np.zeros(3) # Current color.
O = np.array([0., 2.70, 1.57]) # Camera.
Q = np.array([0., -1.1, 2.35]) # Camera pointing to.

img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-4., -4. / r + .25, 4., 4. / r + .25)

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # Loop through initial and secondary rays.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('P3.png', img)