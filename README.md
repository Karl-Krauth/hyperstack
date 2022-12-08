# hyperstack: numpy arrays for microscopy
A Python package for manipulating microscopy images formatted as
[ImageJ-style hyperstacks](https://imagej.nih.gov/ij/docs/guide/146-8.html).

Hyperstack allows you to easily read your image data from files and interact
with them as numpy arrays.
```python
>>> import hyperstack as hs
>>> images = hs.read_ome("example.ome.tif")
>>> images.dims
'tcyx'
>>> images.channels
['brightfield', 'cy5', 'egfp']
>>> images.shape
(5, 3, 1024, 1024)
>>> images[0, 'cy5']
array([[121, 114, 101, ..., 124, 120, 132],
       [127, 120, 125, ..., 170, 143, 135],
       [134, 130, 109, ..., 178, 142, 137],
       ...,
       [120, 114, 154, ..., 143, 165, 141],
       [127, 129, 156, ..., 162, 171, 121],
       [128, 123, 131, ..., 159, 177, 136]], dtype=uint16)
>>> images.show(0, 'cy5')
TODO: image here
```

Currently hyperstack can read OME-Tiff files generated by
[μManager 2.0](https://micro-manager.org/Micro-Manager_File_Formats).
Pull requests adding support for other file formats are welcome.
