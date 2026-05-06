### FashionCloth-v1
#### Overview
FashionCloth includes 189,966 garments.For each garment,there is a corresponding sketch.For different tasks, we divide different datasets.For the three tasks of clothing synthesis, logo customization, and texture personalization,the dataset has 189,966,  11,632 and 45,317 image pairs, respectively.And we provide 1,267 image pairs for test.

#### Data Structure
The dataset includes cloth(189,966 images in folder'./cloth'),sketch(189,966 images in folder'./sketch'),logo(11,632 images in folder'./logo'),mask(11,632 images in folder'./mask'),texture(45,317 images in folder'./texture'),test data(in folder'./test_data'),sketch annotations(in file'./sketch_pair.json'),logo annotations(in file'./logo_pair.json') and texture annotations(in file'./.texture_pair.json').Under the test_data folder, there are four folders, each containing 1,267 images.
```
|- ./FashionCloth-v1   
    |- cloth
    |   |- cloth_000000.jpg
    |   |- .....
    |   └  cloth_189965.jpg
    |- sketch 
    |   |- sketch_000000.jpg
    |   |- ......
    |   └ sketch_189965.jpg
    |- logo
    |   |- 1ogo_000024.jpg
    |   |- ......
    |   └  logo_148678.jpg
    |- mask
    |   |- mask_000024.jpg
    |   |- ......
    |   └  mask_148678.jpg
    |- texture
    |   |- texture_000009.jpg
    |   |- ......
    |   └  texture_120313.jpg
    |- test_data
    |   |- cloth
    |   |- logo
    |   |- mask
    |   |- sketch
    |   └  test_pair.json
    |- sketch_pair.json
    |- logo_pair.json
    |- masked_logo_pair.json
    └  texture_pair.json
```
##### sketch annotation example:
```
{
    "cloth": "cloth_path"
    "sketch": "sketch_path"
    "caption": "caption"
}
```
##### logo annotation example:
```
{
    "cloth": "cloth_path",
    "sketch": "sketch_path",
    "logo": "logo_path",
    "mask": "mask_path",
    "caption": "caption"
}
```
##### texture annotation example:
```
{
    "cloth": "cloth_path",
    "sketch": "sketch_path",
    "texture": "texture_path",
    "caption": "caption"
}
```

