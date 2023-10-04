# Como ejecutar esto?

Foobar is a Python library for dealing with word pluralization.

## Descargar el rosbag y colocarlos en la carpera resources

```bash
cd resource/rosbag2
```

y guardar el rosbag de la maeteria 'rosbag2_2022_11_09-15_21_22_0.db3'

## Lanzar los nodos necesarios

Lanzar el rosbag
En una consola

```bash
. source.sh
./rosbag.sh
```

Lanzar el nodo estereo image proc
En otra consola

```bash
. source.sh
./stereo.sh
```

## Lanzar el script del TP necesario

Ej

```bash
. source.sh
python ./src.imgstereo_pt08.py
```

## Ver los resultados

Usar el RVIZ para ver los resultados

```bash
. source.sh
rviz2
```
### Ver los resultados

Info adicional para buscar los resulados
- KeyPoins Left: /left/kp
- KeyPoins Right: /right/kp
- Matches: /stereo/matches
- Nube de puntos: /stereo/pointcloud