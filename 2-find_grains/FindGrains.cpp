#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tiffio.h>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

struct zone_1g2e {
  int xTL, xTR, xBR, xBL;
  int yTL, yTR, yBR, yBL;
  int Rcorners;
};

struct distPoint {
  int dx, dy;
  float dist;
  distPoint(int DX, int DY) : dx(DX), dy(DY) { dist = sqrt(dx * dx + dy * dy); }
};

struct grain {
  int x, y, radius;  // all expressed in pixel units
  grain() : x(0), y(0), radius(0) {}
  grain(int X, int Y, int R) : x(X), y(Y), radius(R) {}
};

struct Data {
  vector<vector<unsigned short> > image;
  int dimx, dimy;
  zone_1g2e zone;

  string filename;
  int Rmax;
  int Rmin;
  int nsmooth;
  unsigned short threshold;
};

inline int nearest(double x) { return (int)floor(x + 0.5); }

class thumbnail {
 public:
  vector<vector<unsigned short> > imThumb;
  int imW;
  int imH;
  int image_div;

  // Ctor
  thumbnail(vector<vector<unsigned short> >& image, int dimx, int dimy, int imageDiv) : image_div(imageDiv) {
    imW = dimx / image_div;  // integer division
    imH = dimy / image_div;

    imThumb.resize(imW);
    for (int i = 0; i < imW; i++) {
      imThumb[i].resize(imH);
    }

    // Resize
    int x0 = 0, y0 = 0;
    unsigned short mean_grey;
    double fact;
    int ix = 0, iy = 0;

    for (int xm = image_div; xm < dimx; xm += image_div) {
      y0 = 0;
      iy = 0;
      for (int ym = image_div; ym < dimy; ym += image_div) {
        mean_grey = 0;
        fact = 1.0 / ((xm - x0) * (ym - y0));
        for (int x = x0; x < xm; x++) {
          for (int y = y0; y < ym; y++) {
            mean_grey += nearest(fact * image[x][y]);
          }
        }
        imThumb[ix][iy] = mean_grey;

        iy++;
        if (iy >= imH) iy = imH - 1;

        y0 += image_div;
      }

      x0 += image_div;
      ix++;
      if (ix >= imW) ix = imW - 1;
    }
  }

  void writePgm(const char* name) {
    ofstream file(name, ios::binary);
    file << "P5" << endl;
    file << imW << " " << imH << endl;
    file << "255" << endl;

    double conv = 255.0 / 65535.0;

    unsigned char v;
    for (int y = 0; y < imH; y++) {
      for (int x = 0; x < imW; x++) {
        v = (unsigned char)nearest(conv * (imThumb[x][y]));
        file.write((const char*)&v, sizeof(unsigned char));
      }
    }
  }
};

template <class T>
void draw_disk(vector<vector<T> >& image, int dimx, int dimy, int x_centre, int y_centre, int radius, T col) {
  int d = 3 - (2 * radius);
  int x = 0;
  int y = radius;

  while (y >= x) {
    for (int i = x_centre - x; i <= x_centre - x + 2 * x + 1; i++) {
      int xx = i;
      int yy = y_centre - y;
      if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;
    }
    for (int i = x_centre - x; i <= x_centre - x + 2 * x + 1; i++) {
      int xx = i;
      int yy = y_centre + y;
      if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;
    }
    for (int i = x_centre - y; i <= x_centre - y + 2 * y + 1; i++) {
      int xx = i;
      int yy = y_centre - x;
      if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;
    }
    for (int i = x_centre - y; i <= x_centre - y + 2 * y + 1; i++) {
      int xx = i;
      int yy = y_centre + x;
      if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;
    }

    if (d < 0)
      d = d + (4 * x) + 6;
    else {
      d = d + 4 * (x - y) + 10;
      y--;
    }

    x++;
  }
}

template <class T>
void draw_circle(vector<vector<T> >& image, int dimx, int dimy, int x_centre, int y_centre, int radius, T col) {
  int x = 0, y = radius, m = 5 - 4 * radius;
  int xx, yy;
  while (x <= y) {  /// @see http://fr.wikipedia.org/wiki/Algorithme_de_tracé_d'arc_de_cercle_de_Bresenham
    xx = x + x_centre;
    yy = y + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = y + x_centre;
    yy = x + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = -x + x_centre;
    yy = y + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = -y + x_centre;
    yy = x + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = x + x_centre;
    yy = -y + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = y + x_centre;
    yy = -x + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = -x + x_centre;
    yy = -y + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    xx = -y + x_centre;
    yy = -x + y_centre;
    if (xx >= 0 && xx < dimx && yy >= 0 && yy < dimy) image[xx][yy] = col;

    if (m > 0) {
      y = y - 1;
      m = m - 8 * y;
    }
    x = x + 1;
    m = m + 8 * x + 4;
  }
}

template <class T>
void draw_triangle(vector<vector<T> >& image, int dimx, int dimy, int X1, int Y1, int X2, int Y2, int X3, int Y3,
                   T Col1, T Col2, T Col3) {
  // numbering
  int x0, y0;
  int x1, y1;
  int x2, y2;
  int x3, y3;
  T col1, col2, col3;

  y3 = std::max(Y1, Y2);
  y3 = std::max(y3, Y3);
  if (y3 == Y1) {
    x3 = X1;
    col3 = Col1;
    y2 = std::min(Y3, Y2);
    if (y2 == Y3) {
      x2 = X3;
      col2 = Col3;
      x1 = X2;
      y1 = Y2;
      col1 = Col2;
    } else {
      x2 = X2;
      col2 = Col2;
      x1 = X3;
      y1 = Y3;
      col1 = Col3;
    }
  } else if (y3 == Y2) {
    x3 = X2;
    col3 = Col2;
    y2 = std::min(Y3, Y1);
    if (y2 == Y3) {
      x2 = X3;
      col2 = Col3;
      x1 = X1;
      y1 = Y1;
      col1 = Col1;
    } else {
      x2 = X1;
      col2 = Col1;
      x1 = X3;
      y1 = Y3;
      col1 = Col3;
    }
  } else {
    x3 = X3;
    col3 = Col3;
    y2 = std::min(Y2, Y1);
    if (y2 == Y2) {
      x2 = X2;
      col2 = Col2;
      x1 = X1;
      y1 = Y1;
      col1 = Col1;
    } else {
      x2 = X1;
      col2 = Col1;
      x1 = X2;
      y1 = Y2;
      col1 = Col2;
    }
  }

  x0 = (x3 - x2) * (y1 - y2) / (y3 - y2) + x2;
  y0 = y1;

  double dx = x0 - x2;
  double dy = y0 - y2;
  double lx = x3 - x2;
  double ly = y3 - y2;
  double l = sqrt((dx * dx + dy * dy) / (lx * lx + ly * ly));
  T col0 = (1.0 - l) * col2 + l * col3;

  // first half
  int H = y0 - y2;
  double invH = 1.0 / (double)H;
  for (int y = 0; y <= H; y++) {
    double l = (double)y * invH;
    int xd = nearest((1.0 - l) * x2 + l * x0);
    int xe = nearest((1.0 - l) * x2 + l * x1);
    T cold = (1.0 - l) * col2 + l * col0;
    T cole = (1.0 - l) * col2 + l * col1;
    if (xd > xe) {
      std::swap(xd, xe);
      std::swap(cold, cole);
    }
    double invL = 1.0 / (double)(xe - xd);
    for (int x = xd; x <= xe; x++) {
      double l2 = (x - xd) * invL;
      T col = (1.0 - l2) * cold + l2 * cole;
      int yy = y2 + y;
      if (x >= 0 && x < dimx && yy >= 0 && yy < dimy) image[x][yy] = col;
    }
  }

  // second half
  H = y3 - y0;
  invH = 1.0 / (double)H;
  for (int y = 1; y <= H; y++) {
    double l = (double)y * invH;
    int xd = nearest((1.0 - l) * x3 + l * x0);
    int xe = nearest((1.0 - l) * x3 + l * x1);
    T cold = (1.0 - l) * col3 + l * col0;
    T cole = (1.0 - l) * col3 + l * col1;
    if (xd > xe) {
      std::swap(xd, xe);
      std::swap(cold, cole);
    }
    double invL = 1.0 / (double)(xe - xd);
    for (int x = xd; x <= xe; x++) {
      double l2 = (x - xd) * invL;
      T col = (1.0 - l2) * cold + l2 * cole;
      int yy = y3 - y;
      if (x >= 0 && x < dimx && yy >= 0 && yy < dimy) image[x][yy] = col;
    }
  }
}

// Process has done i out of n rounds,
// and we want a bar of width w and resolution r.
static inline void loadbar(size_t x, size_t n, size_t w = 50) {
  if ((x != n) && (x % (n / 100 + 1) != 0)) return;

  float ratio = x / (float)n;
  size_t c = ratio * w;

  cerr << setw(3) << (size_t)(ratio * 100) << "% [";
  for (size_t x = 0; x < c; x++) cerr << "|";
  for (size_t x = c; x < w; x++) cerr << " ";
  cerr << "]\r" << flush;
}

// Read tiff image (16-bits Gray levels)
void readTiff(const char* name, vector<vector<unsigned short> >& image, int& dimx, int& dimy) {
  cout << "-> Read " << name << endl;
  TIFF* tif = TIFFOpen(name, "r");

  if (!tif) {
    cerr << "Cannot open tiff file named '" << name << "'" << endl;
    exit(0);
  }
  uint32 w, h;
  size_t npixels;
  uint32* raster;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
  cout << w << "x" << h << endl;
  npixels = w * h;

  raster = (uint32*)_TIFFmalloc(npixels * sizeof(uint32));
  if (raster != NULL) {
    if (!TIFFReadRGBAImage(tif, w, h, raster, 0)) {
      cerr << "Cannot read tiff file named '" << name << "'" << endl;
      exit(0);
    }
  }

  dimx = (int)w;
  dimy = (int)h;

  // Reserve memory for image
  if (!image.empty()) image.clear();
  image.resize(dimx);

  for (int ix = 0; ix < dimx; ix++) {
    image[ix].resize(dimy);
  }

  // unsigned short r, g, b;
  double r, g, b;
  // int dimxy = dimx * dimy;
  for (int row = 0; row < dimy; ++row) {
    int shift = row * dimx;
    for (int col = 0; col < dimx; ++col) {
      //			r = (unsigned short) raster[dimxy - (shift + (dimx-x))    ];
      //			g = (unsigned short) raster[dimxy - (shift + (dimx-x)) + 1];
      //			b = (unsigned short) raster[dimxy - (shift + (dimx-x)) + 2];
      int p = shift + col;
      r = TIFFGetR(raster[p]) / 255.;
      g = TIFFGetG(raster[p]) / 255.;
      b = TIFFGetB(raster[p]) / 255.;

      image[col][dimy - row - 1] = (unsigned short)(65535.0 * (0.299 * r + 0.587 * g + 0.114 * b));
    }
  }

  _TIFFfree(raster);
  TIFFClose(tif);
}

void zoning(zone_1g2e& zone, vector<vector<unsigned short> >& image, int dimx, int dimy) {
  cout << "-> Zoning" << endl;
  int vx, vy, lim;
  int dx, dy;

  // top
  vx = zone.xTR - zone.xTL;
  vy = zone.yTR - zone.yTL;
  lim = std::max(zone.yTL, zone.yTR);
  for (int y = 0; y < lim; ++y) {
    for (int x = 0; x < dimx; ++x) {
      dx = x - zone.xTL;
      dy = y - zone.yTL;
      if (vx * dy - vy * dx < 0) image[x][y] = 0;
    }
  }

  // right
  vx = zone.xBR - zone.xTR;
  vy = zone.yBR - zone.yTR;
  lim = std::min(zone.xTR, zone.xBR);
  for (int y = 0; y < dimy; ++y) {
    for (int x = lim; x < dimx; ++x) {
      dx = x - zone.xTR;
      dy = y - zone.yTR;
      if (vx * dy - vy * dx < 0) image[x][y] = 0;
    }
  }

  // bottom
  vx = zone.xBL - zone.xBR;
  vy = zone.yBL - zone.yBR;
  lim = std::min(zone.yBL, zone.yBR);
  for (int y = lim; y < dimy; ++y) {
    for (int x = 0; x < dimx; ++x) {
      dx = x - zone.xBR;
      dy = y - zone.yBR;
      if (vx * dy - vy * dx < 0) image[x][y] = 0;
    }
  }

  // left
  vx = zone.xTL - zone.xBL;
  vy = zone.yTL - zone.yBL;
  lim = std::max(zone.xTL, zone.xBL);
  for (int y = 0; y < dimy; ++y) {
    for (int x = 0; x < lim; ++x) {
      dx = x - zone.xBL;
      dy = y - zone.yBL;
      if (vx * dy - vy * dx < 0) image[x][y] = 0;
    }
  }

  // Corner disks
  draw_disk<unsigned short>(image, dimx, dimy, zone.xTL, zone.yTL, zone.Rcorners, 0);
  draw_disk<unsigned short>(image, dimx, dimy, zone.xTR, zone.yTR, zone.Rcorners, 0);
  draw_disk<unsigned short>(image, dimx, dimy, zone.xBR, zone.yBR, zone.Rcorners, 0);
  draw_disk<unsigned short>(image, dimx, dimy, zone.xBL, zone.yBL, zone.Rcorners, 0);
}

void histogram(vector<vector<unsigned short> >& image, int dimx, int dimy) {
  cout << "-> Histogram" << endl;
  vector<size_t> hist(65535);
  for (size_t i = 0; i < 65535; i++) {
    hist[i] = 0;
  }

  for (int ix = 0; ix < dimx; ix++) {
    for (int iy = 0; iy < dimy; iy++) {
      hist[image[ix][iy]]++;
    }
    loadbar(ix + 1, dimx);
  }
  cout << endl;

  ofstream file("histo.txt");
  file << "0" << endl;  // set to zero to avoid the peak at 0
  for (size_t i = 1; i < 65535; i++) {
    file << hist[i] << endl;
  }
  cout << "Histogram saved in 'hist.txt'" << endl;
}

void segment(vector<vector<unsigned short> >& image, int dimx, int dimy, unsigned short threshold) {
  cout << "-> Segmentation" << endl;
  for (int ix = 0; ix < dimx; ix++) {
    for (int iy = 0; iy < dimy; iy++) {
      if (image[ix][iy] > threshold)
        image[ix][iy] = 65535;
      else
        image[ix][iy] = 0;
    }
    loadbar(ix + 1, dimx);
  }
  cout << endl;
}

void smooth(vector<vector<unsigned short> >& image, int dimx, int dimy) {
  cout << "-> Smoothing" << endl;

  vector<vector<unsigned short> > imageS;
  imageS.resize(dimx);
  for (int ix = 0; ix < dimx; ix++) {
    imageS[ix].resize(dimy);
  }

  for (int ix = 0; ix < dimx; ix++) {
    for (int iy = 0; iy < dimy; iy++) {
      imageS[ix][iy] = image[ix][iy];
    }
  }

  for (int ix = 2; ix < dimx - 2; ix++) {
    for (int iy = 2; iy < dimy - 2; iy++) {
      image[ix][iy] = nearest(
          0.003663003663 *
          (image[ix - 2][iy - 2] + 4 * image[ix - 1][iy - 2] + 7 * image[ix][iy - 2] + 4 * image[ix + 1][iy - 2] +
           image[ix + 2][iy - 2] + 4 * image[ix - 2][iy - 1] + 16 * image[ix - 1][iy - 1] + 26 * image[ix][iy - 1] +
           16 * image[ix + 1][iy - 1] + 4 * image[ix + 2][iy - 1] + 7 * image[ix - 2][iy] + 26 * image[ix - 1][iy] +
           41 * image[ix][iy] + 26 * image[ix + 1][iy] + 7 * image[ix + 2][iy] + 4 * image[ix - 2][iy + 1] +
           16 * image[ix - 1][iy + 1] + 26 * image[ix][iy + 1] + 16 * image[ix + 1][iy + 1] +
           4 * image[ix + 2][iy + 1] + image[ix - 2][iy + 2] + 4 * image[ix - 1][iy + 2] + 7 * image[ix][iy + 2] +
           4 * image[ix + 1][iy + 2] + image[ix + 2][iy + 2]));
    }
    loadbar(ix + 1, dimx - 4);
  }
  cout << endl;
}

void distance(vector<vector<unsigned short> >& image, vector<vector<float> >& distMap, int dimx, int dimy,
              int layer_max) {
  cout << "-> Distance Map" << endl;

  // masque concentric (algorithm d'Andres pour eviter les trous)
  vector<distPoint> distPoints;
  for (int r = 1; r <= layer_max; r++) {
    int x = 0;
    int y = r;
    int d = r - 1;
    while (y >= x) {
      distPoints.push_back(distPoint(x, y));
      distPoints.push_back(distPoint(y, x));
      distPoints.push_back(distPoint(-x, y));
      distPoints.push_back(distPoint(-y, x));
      distPoints.push_back(distPoint(x, -y));
      distPoints.push_back(distPoint(y, -x));
      distPoints.push_back(distPoint(-x, -y));
      distPoints.push_back(distPoint(-y, -x));
      if (d >= 2 * x) {
        d = d - 2 * x - 1;
        x = x + 1;
      } else if (d < 2 * (r - y)) {
        d = d + 2 * y - 1;
        y = y - 1;
      } else {
        d = d + 2 * (y - x - 1);
        y = y - 1;
        x = x + 1;
      }
    }
  }

  distMap.resize(dimx);
  for (int ix = 0; ix < dimx; ix++) {
    distMap[ix].resize(dimy);
  }

  int sum = 0;

#pragma omp parallel for shared(sum)
  for (int ix = 0; ix < dimx; ix++) {
    for (int iy = 0; iy < dimy; iy++) {
      if (image[ix][iy] == 0) {
        distMap[ix][iy] = 0.0;
        continue;
      }
      distMap[ix][iy] = 0.0;

      for (size_t i = 0; i < distPoints.size(); i++) {
        int x = ix + distPoints[i].dx;
        if (x < 0 || x >= dimx) continue;
        int y = iy + distPoints[i].dy;
        if (y < 0 || y >= dimy) continue;

        if (image[x][y] == 0) {
          distMap[ix][iy] = distPoints[i].dist;
          break;
        }
      }
    }
#pragma omp atomic
    sum += 1;

#pragma omp critical
    { loadbar(sum + 1, dimx); }
  }
  cout << endl;
}

void extract(vector<vector<float> >& distMap, int dimx, int dimy, vector<grain>& grains, int Rmax, int Rmin) {
  cout << "-> Extract grain positions and radii" << endl;

  float Rup = (float)Rmax;
  float Rdown = (float)(Rmax - 1);

  int count = 0;
  while (Rdown >= (float)Rmin) {
    for (int ix = 0; ix < dimx; ix++) {
      for (int iy = 0; iy < dimy; iy++) {
        if (distMap[ix][iy] <= Rup && distMap[ix][iy] > Rdown) {

          int radius = nearest(distMap[ix][iy]);
          int x_centre = ix;
          int y_centre = iy;

          grains.push_back(grain(x_centre, y_centre, radius));
          radius += (Rmin - 1);  // on masque un peu plus que le grain lui meme
          draw_disk<float>(distMap, dimx, dimy, x_centre, y_centre, radius, 0.0);
        }
      }
    }

    loadbar(count++, Rmax - Rmin - 2);
    Rup -= 1.0;
    Rdown -= 1.0;
  }
  cout << endl;
}

void superImpose(const char* name, const char* nameOut, vector<grain>& grains) {
  cout << "-> Super-impose grains over the image for checking" << endl;

  int dimx, dimy;
  vector<vector<unsigned short> > image;

  readTiff(name, image, dimx, dimy);

  vector<vector<unsigned char> > sample;
  sample.resize(dimx);
  for (int ix = 0; ix < dimx; ix++) {
    sample[ix].resize(dimy);
  }

  for (size_t i = 0; i < grains.size(); i++) {
    draw_disk<unsigned char>(sample, dimx, dimy, grains[i].x, grains[i].y, grains[i].radius, 155);
    draw_circle<unsigned char>(sample, dimx, dimy, grains[i].x, grains[i].y, grains[i].radius, 255);
    loadbar(i + 1, grains.size());
  }
  cout << endl;

  // Merge images
  ofstream file(nameOut, ios::binary);
  file << "P6" << endl;
  file << dimx << " " << dimy << endl;
  file << "255" << endl;

  double conv = 255.0 / 65535.0;
  double alpha = 1.0;  // 0.4;

  unsigned char v, r, g, b;
  for (int y = 0; y < dimy; y++) {
    for (int x = 0; x < dimx; x++) {
      v = (unsigned char)nearest(conv * (image[x][y]));
      r = b = v;
      if (sample[x][y] > 0)
        g = (unsigned char)nearest(alpha * sample[x][y] + (1.0 - alpha) * v);
      else
        g = v;
      file.write((const char*)&r, sizeof(unsigned char));
      file.write((const char*)&g, sizeof(unsigned char));
      file.write((const char*)&b, sizeof(unsigned char));
    }
  }
}

void read_data(const char* name, Data& data) {
  ifstream dataFile(name);
  if (!dataFile) {
    cerr << "Cannot open file " << name << endl;
    exit(0);
  }

  string token;
  dataFile >> token;

  while (dataFile) {

    if (token[0] == '!')
      getline(dataFile, token);
    else if (token == "image_name") {
      dataFile >> data.filename;
    } else if (token == "Rmin") {
      dataFile >> data.Rmin;
    } else if (token == "Rmax") {
      dataFile >> data.Rmax;
    } else if (token == "nsmooth") {
      dataFile >> data.nsmooth;
    } else if (token == "zone.xTL") {
      dataFile >> data.zone.xTL;
    } else if (token == "zone.yTL") {
      dataFile >> data.zone.yTL;
    } else if (token == "zone.xTR") {
      dataFile >> data.zone.xTR;
    } else if (token == "zone.yTR") {
      dataFile >> data.zone.yTR;
    } else if (token == "zone.xBR") {
      dataFile >> data.zone.xBR;
    } else if (token == "zone.yBR") {
      dataFile >> data.zone.yBR;
    } else if (token == "zone.xBL") {
      dataFile >> data.zone.xBL;
    } else if (token == "zone.yBL") {
      dataFile >> data.zone.yBL;
    } else if (token == "zone.Rcorners") {
      dataFile >> data.zone.Rcorners;
    } else if (token == "threshold") {
      dataFile >> data.threshold;
    } else {
      fprintf(stdout, "Unknown token: %s\n", token.c_str());
    }

    dataFile >> token;
  }
}

void clean(const char* GrainsFileName, const char* CommandFileName, const char* ImageFileName) {
  cout << "-> Cleaning" << endl;
  // Read Grains
  ifstream Gfile(GrainsFileName);
  if (!Gfile) {
    cerr << "Cannot open file " << GrainsFileName << endl;
    exit(0);
  }

  vector<grain> grains;
  int nbGrains;
  Gfile >> nbGrains;
  grain G;
  int unused;
  for (int i = 0; i < nbGrains; i++) {
    Gfile >> G.x >> G.y >> unused >> G.radius;
    grains.push_back(G);
  }
  Gfile.close();

  // Read Command
  ifstream Cfile(CommandFileName);
  if (!Cfile) {
    cerr << "Cannot open file " << CommandFileName << endl;
    exit(0);
  }

  string token;
  Cfile >> token;
  vector<grain> grains_remove;
  vector<grain> grains_add;
  G.radius = 1;
  while (Cfile) {
    if (token == "remove") {
      int nb;
      Cfile >> nb;
      for (int n = 0; n < nb; n++) {
        Cfile >> G.x >> G.y;
        grains_remove.push_back(G);
      }
    } else if (token == "add") {
      int nb;
      Cfile >> nb;
      for (int n = 0; n < nb; n++) {
        Cfile >> G.x >> G.y >> G.radius;
        cout << G.x << endl;
        grains_add.push_back(G);
      }
    } else
      cerr << "Unknown keyword: " << token << endl;

    Cfile >> token;
  }
  Cfile.close();

  // Remove
  for (size_t ir = 0; ir < grains_remove.size(); ir++) {
    for (size_t i = 0; i < grains.size(); i++) {
      double dx = grains_remove[ir].x - grains[i].x;
      double dy = grains_remove[ir].y - grains[i].y;
      double d = sqrt(dx * dx + dy * dy);
      if (d < grains[i].radius) grains[i].radius = 0;
    }
  }
  vector<grain> grains2;
  for (size_t i = 0; i < grains.size(); i++) {
    if (grains[i].radius > 0) grains2.push_back(grains[i]);
  }

  // Add
  for (size_t i = 0; i < grains_add.size(); i++) {
    grains2.push_back(grains_add[i]);
  }

  // Store
  ofstream fg("Cleaned.txt");
  fg << grains2.size() << endl << endl;
  for (size_t i = 0; i < grains2.size(); i++) {
    fg << grains2[i].x << " " << grains2[i].y << " 0 " << grains2[i].radius << endl;
  }

  // Image for check
  superImpose(ImageFileName, "Cleaned.ppm", grains2);
}

int main(int argc, char const* argv[]) {
  Data data;

  if (argc == 2) {
    read_data(argv[1], data);
  } else if (argc == 4) {
    clean(argv[1], argv[2], argv[3]);
    exit(0);
  } else {
    cerr << "Usages: \n" << argv[0] << " InputDataFile" << endl;
    cerr << "or\n" << argv[0] << " GrainDataFile CleaningCommands ImageName.tif" << endl << endl;
    exit(0);
  }

  readTiff(data.filename.c_str(), data.image, data.dimx, data.dimy);

  zoning(data.zone, data.image, data.dimx, data.dimy);
  if (1) {
    thumbnail thumb(data.image, data.dimx, data.dimy, 1);
    thumb.writePgm("zoning.pgm");
  }

  histogram(data.image, data.dimx, data.dimy);
  if (data.threshold == 0) {
    cerr << "The file histo.txt has been created. Please use it to set a non-zero value to threshold." << endl;
    return 0;
  }

  for (int n = 0; n < data.nsmooth; n++) smooth(data.image, data.dimx, data.dimy);
  if (1) {
    thumbnail thumb(data.image, data.dimx, data.dimy, 1);
    thumb.writePgm("smooth.pgm");
  }

  segment(data.image, data.dimx, data.dimy, data.threshold);
  if (1) {
    thumbnail thumb(data.image, data.dimx, data.dimy, 1);
    thumb.writePgm("segment.pgm");
  }

  vector<vector<float> > distMap;
  int dist_max = nearest(1.1 * data.Rmax);
  distance(data.image, distMap, data.dimx, data.dimy, dist_max);
  if (1) {
    double rescalFactor = 65535.0 / (sqrt(2.0) * dist_max + 1.0);
    for (int ix = 0; ix < data.dimx; ix++) {
      for (int iy = 0; iy < data.dimy; iy++) {
        data.image[ix][iy] = nearest(rescalFactor * distMap[ix][iy]);
      }
    }
    thumbnail thumb(data.image, data.dimx, data.dimy, 1);
    thumb.writePgm("distance.pgm");
  }

  vector<grain> grains;
  extract(distMap, data.dimx, data.dimy, grains, data.Rmax, data.Rmin);

  // Image for check
  superImpose(data.filename.c_str(), "Grains.ppm", grains);

  // save
  ofstream fg("grains.txt");
  fg << grains.size() << endl << endl;
  for (size_t i = 0; i < grains.size(); i++) {
    fg << grains[i].x << "\t" << grains[i].y << "\t0\t" << grains[i].radius << endl;
  }

  return 0;
}
