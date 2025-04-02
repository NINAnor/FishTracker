{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312.withPackages (ps: with ps; [
    uv
    pyqt5
    matplotlib
    numpy
    opencv4
    pytz
    scikit-image
    seaborn
    scikit-learn
    numba
    msgpack
    filterpy
  ]);
in
pkgs.mkShell {
  buildInputs = [
    python
    pkgs.qt5.full

    # GUI/graphics dependencies
    pkgs.xorg.libxcb
    pkgs.libxkbcommon
    pkgs.xorg.libXcursor
    pkgs.xorg.libXrandr
    pkgs.xorg.libXrender
    pkgs.xorg.libXi
    pkgs.xorg.libXext
    pkgs.xorg.libSM
    pkgs.tk
    pkgs.fontconfig
    pkgs.freetype
    pkgs.dbus
    pkgs.glib
    pkgs.libGL
    pkgs.libstdcxx5
  ];

  shellHook = ''
    export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.full}/${pkgs.qt5.qtbase.qtPluginPrefix}/platforms
    export QT_PLUGIN_PATH=${pkgs.qt5.full}/${pkgs.qt5.qtbase.qtPluginPrefix}
    export QT_DEBUG_PLUGINS=1
    echo "🐟 FishTracker's full environment is ready!"
    echo "Run 'python main.py' to start the application."
  '';
}
