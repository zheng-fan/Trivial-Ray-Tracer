This is a trivial 3D rendering engine written in C++.

# Features

* Phong Model and Monte Carlo Path Tracing Model
* Rendering of plane, sphere and mesh
* Read mesh from a obj file
* Custom texture from a image file
* Bounding box acceleration
* Parallel acceleration: both MPI and OpenACC

# Dependencies

* GLM
* FreeImage

You may need command like this to install FreeImage or just compile it yourself:

```
sudo apt install libfreeimage-dev
```

# Demo

A simple demo:

![demo](https://github.com/fz568573448/Simple-Ray-Tracer/blob/master/demo/demo.jpg)
