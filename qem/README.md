# FA-QEM Mesh Simplifier

A C++17 command-line tool for reducing a polygon mesh using feature-aware quadric error metrics (FA-QEM). It uses CGAL for mesh data structures and mesh operations. OpenCV is optional and enables an additional multi-view silhouette cost term.

## 1. Requirements

### Required

- A C++17-compatible compiler
- CMake 3.15 or newer
- CGAL development files
- Make or a supported build tool on Linux

### Optional: multi-view silhouette preservation

- OpenCV with the `core`, `imgproc`, and `imgcodecs` components
- A COLMAP text model containing `cameras.txt` and `images.txt`
- One binary silhouette mask per image/view

**OpenCV is optional.** Without it, the program still builds and performs mesh simplification, but the silhouette term is disabled. COLMAP itself is not required to compile or run the basic simplifier; it is only needed to generate the camera text files used by the optional silhouette feature.

---

## 2. Install dependencies

Use the section for your operating system.

### Fedora

Install the compiler, CMake, CGAL, and OpenCV development packages:

```bash
sudo dnf install -y gcc-c++ cmake make CGAL-devel opencv-devel
```

For basic mesh simplification only, OpenCV can be omitted:

```bash
sudo dnf install -y gcc-c++ cmake make CGAL-devel
```

### Arch Linux

```bash
sudo pacman -Syu --needed base-devel cmake cgal opencv
```

`base-devel` provides the standard build tools, including the compiler and Make. For basic simplification without silhouette masks, omit `opencv`:

```bash
sudo pacman -Syu --needed base-devel cmake cgal
```

### Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential cmake libcgal-dev libopencv-dev
```

For basic simplification without silhouette masks, omit `libopencv-dev`:

```bash
sudo apt update
sudo apt install -y build-essential cmake libcgal-dev
```

### Windows 10/11 (Visual Studio + vcpkg)

The recommended Windows route is MSVC (Visual Studio 2022) with vcpkg managing CGAL and its dependencies.

1. Install **Visual Studio 2022 Community** or **Build Tools for Visual Studio 2022**. In the Visual Studio Installer, select the **Desktop development with C++** workload. Make sure the MSVC x64/x86 build tools and a Windows SDK are selected.
2. Install CMake and Git if they are not already available in your terminal. With `winget`:

   ```powershell
   winget install --id Kitware.CMake -e
   winget install --id Git.Git -e
   ```

3. Open a new PowerShell window and clone/bootstrap vcpkg:

   ```powershell
   git clone https://github.com/microsoft/vcpkg C:\dev\vcpkg
   cd C:\dev\vcpkg
   .\bootstrap-vcpkg.bat
   ```

4. Install CGAL for 64-bit Windows. The extra 32-bit `yasm-tool` install is a workaround documented by CGAL for building the 64-bit GMP dependency through vcpkg:

   ```powershell
   .\vcpkg.exe install yasm-tool:x86-windows
   .\vcpkg.exe install cgal:x64-windows
   ```

5. **Optional — install OpenCV** to enable the silhouette feature:

   ```powershell
   .\vcpkg.exe install opencv:x64-windows
   ```

   Skip this step for basic mesh simplification. The CMake project will detect that OpenCV is missing and build with the silhouette term disabled.

If vcpkg is installed somewhere other than `C:\dev\vcpkg`, replace that path in the CMake command below.

---

## 3. Build the project

Run these commands from the directory containing `CMakeLists.txt`.

### Linux (Fedora, Arch, Ubuntu)

Configure and compile in Release mode:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable will be created at:

```text
build/lowpoly
```

If CMake reports that OpenCV was not found, the build can still succeed. The executable will print that the silhouette feature is disabled when run without OpenCV support.

### Windows (PowerShell)

From the project directory, configure with the vcpkg toolchain and build the Release configuration:

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel
```

With the default Visual Studio generator, the executable will normally be at:

```text
build\Release\lowpoly.exe
```

If you installed vcpkg or the project in a different location, update the paths accordingly. If CMake cannot find CGAL, check that the vcpkg toolchain file points to the same vcpkg installation where `cgal:x64-windows` was installed.

---

## 4. Run the program

The program takes an input mesh and an output mesh as its first two arguments. It reads and writes formats supported by CGAL's polygon-mesh I/O; the supplied sample is a PLY file.

### Linux example

Run from the project directory:

```bash
./build/lowpoly out_poisson.ply simplified.ply --faces 5000
```

### Windows example

Run from the project directory:

```powershell
.\build\Release\lowpoly.exe .\out_poisson.ply .\simplified.ply --faces 5000
```

This example reads the included `out_poisson.ply` mesh and writes the simplified result to `simplified.ply`. Change the input path to use another mesh.

### Command-line options

| Argument | Meaning | Default |
|---|---|---:|
| `input.ply output.ply` | Input mesh and destination for the simplified mesh | Required |
| `--faces N` | Target number of faces | `5000` |
| `--sharp-angle DEG` | Dihedral-angle threshold used to identify sharp feature edges | `40` |
| `--min-component N` | Minimum connected-component size, in faces, to keep during input cleanup. `0` selects the automatic threshold | `0` |
| `--history FILE` | Write the edge-collapse history to a text file | Not written |
| `--colmap DIR` | Directory containing COLMAP `cameras.txt` and `images.txt` | Not used |
| `--masks DIR` | Directory containing the corresponding silhouette masks | Not used |
| `--lambda-sil X` | Weight of the optional silhouette cost | `1.0` |

The face target is a target, not a guarantee. The simplifier rejects collapses that violate its topology or geometry checks, so it may stop above the requested face count if no valid collapses remain.

---

## 5. Optional: enable multi-view silhouette preservation

This feature requires a build that detected OpenCV, plus a COLMAP text model and silhouette masks.

### 5.1 Export the COLMAP model as text

If your COLMAP reconstruction is in `sparse/0`, convert it to the text format expected by this program:

```bash
colmap model_converter \
  --input_path sparse/0 \
  --output_path colmap_txt \
  --output_type TXT
```

The resulting `colmap_txt` directory must contain:

```text
colmap_txt/
├── cameras.txt
└── images.txt
```

The program reads the camera intrinsics and image poses from these files. Use undistorted images/model for camera models with lens distortion; the current reader warns that distortion parameters are ignored.

### 5.2 Prepare the masks

Place the binary silhouette images in one directory. The mask foreground (object) should be white and the background black. The program tries to match each COLMAP image to a mask using the original image name, `<image-name>.png`, `<image-stem>.png`, or `<image-stem>.jpg`.

For example:

```text
masks/
├── view_01.png
├── view_02.png
└── view_03.png
```

The mask dimensions may differ from the source image dimensions; the code scales mask sampling to the camera image dimensions.

### 5.3 Run with the silhouette term enabled

Linux:

```bash
./build/lowpoly out_poisson.ply simplified_silhouette.ply \
  --faces 5000 \
  --colmap colmap_txt \
  --masks masks \
  --lambda-sil 1.0
```

Windows PowerShell:

```powershell
.\build\Release\lowpoly.exe .\out_poisson.ply .\simplified_silhouette.ply `
  --faces 5000 `
  --colmap .\colmap_txt `
  --masks .\masks `
  --lambda-sil 1.0
```

If OpenCV was not found at build time, or the COLMAP files/masks cannot be loaded, the silhouette term will be disabled or the program will report the relevant error. Check the build output and runtime messages.

---

## 6. Output and notes

- The simplified mesh is written to the output path supplied on the command line.
- `--history FILE` optionally saves the sequence of edge collapses and their recorded costs/positions. This is intended to support later mapping, such as transferring appearance information.
- The ML-related importance and protected-vertex/edge fields are hooks in the current code. No trained ML model is loaded or run by this executable yet.
- The `build/` directory is generated by CMake and should not be committed as source. It can be deleted and regenerated whenever needed.

## References

- [CGAL installation and usage](https://doc.cgal.org/latest/Manual/usage.html)
- [CGAL on Windows with Visual C++ and vcpkg](https://doc.cgal.org/latest/Manual/windows.html)
- [CGAL Windows download guidance](https://www.cgal.org/download/windows.html)
- [vcpkg](https://github.com/microsoft/vcpkg)
