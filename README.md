# Face Recognization is FUN

## TODO

- http://answers.opencv.org/question/5651/face-recognition-with-opencv-24x-accuracy/
- https://github.com/bytefish/facerec
- http://flothesof.github.io/smile-recognition.html
- http://bytefish.de/blog/videofacerec/

## Guidelines

- A proper validation should be applied.  
- Figures is a must!.
- Get as much data as possible, because **Data is king**.
- Don't validate your algorithms on images taken in a very controlled scenario. 


## Design

- Data organization: one folder per person, and one person may have multiple images 

```sh
➜  att_faces git:(refactor) ✗ tree -L 2 | head -n 20
.
├── README
├── s1
│   ├── 1.pgm
│   ├── 2.pgm
│   ├── 3.pgm
...
├── s10
│   ├── 1.pgm
│   ├── 2.pgm
│   ├── 3.pgm
...
```

## Dev Env Setup

```sh
# install opencv 

# for gui and maplotlib integration
pip3 install -U wxPython wxmplot
```

## Good Articles

- [Face rec with deep learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
