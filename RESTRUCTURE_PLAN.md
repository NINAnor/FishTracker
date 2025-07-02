# FishTracker Project Restructuring Plan

## Recommended Directory Structure

```
FishTracker/
├── README.md
├── LICENSE.txt
├── pyproject.toml
├── uv.lock
├── main.py                      # Entry point
├── requirements.txt             # For pip users
├── setup.py                     # For package installation
├── CHANGELOG.md
├── .gitignore
├── .pre-commit-config.yaml
├── configs/                     # Configuration files
│   ├── __init__.py
│   ├── default.yaml
│   └── parameters/
│       ├── detector.yaml
│       ├── tracker.yaml
│       └── filter.yaml
├── fishtracker/                 # Main package
│   ├── __init__.py
│   ├── core/                    # Core business logic
│   │   ├── __init__.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py
│   │   │   ├── background_subtractor.py
│   │   │   └── parameters.py
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── tracker.py
│   │   │   ├── kalman_filter.py
│   │   │   └── parameters.py
│   │   ├── fish/
│   │   │   ├── __init__.py
│   │   │   ├── fish_manager.py
│   │   │   └── fish_list.py
│   │   └── processing/
│   │       ├── __init__.py
│   │       ├── batch_track.py
│   │       ├── track_process.py
│   │       └── sort.py
│   ├── ui/                      # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── widgets/
│   │   │   ├── __init__.py
│   │   │   ├── sonar_widget.py
│   │   │   ├── echogram_widget.py
│   │   │   ├── output_widget.py
│   │   │   ├── detection_list.py
│   │   │   ├── collapsible_box.py
│   │   │   └── zoomable_qlabel.py
│   │   ├── dialogs/
│   │   │   ├── __init__.py
│   │   │   └── batch_dialog.py
│   │   ├── views/
│   │   │   ├── __init__.py
│   │   │   ├── detector_parameters_view.py
│   │   │   ├── tracker_parameters_view.py
│   │   │   └── sonar_view3.py
│   │   └── delegates/
│   │       ├── __init__.py
│   │       └── dropdown_delegate.py
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── file_handler.py
│   │   ├── save_manager.py
│   │   ├── playback_manager.py
│   │   ├── image_manipulation.py
│   │   ├── polar_transform.py
│   │   └── logging.py
│   ├── parameters/              # Parameter management
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── detector_parameters.py
│   │   ├── tracker_parameters.py
│   │   ├── mog_parameters.py
│   │   ├── filter_parameters.py
│   │   └── parameter_list.py
│   └── managers/                # System managers
│       ├── __init__.py
│       ├── ui_manager.py
│       └── user_preferences.py
├── tests/                       # Test files
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_detection/
│   │   ├── __init__.py
│   │   ├── test_detector.py
│   │   └── test_background_subtractor.py
│   ├── test_tracking/
│   │   ├── __init__.py
│   │   └── test_tracker.py
│   └── test_utils/
│       ├── __init__.py
│       └── test_file_handler.py
├── docs/                        # Documentation
│   ├── README.md
│   ├── API.md
│   ├── INSTALLATION.md
│   ├── USER_GUIDE.md
│   └── DEVELOPER_GUIDE.md
├── scripts/                     # Standalone scripts
│   ├── __init__.py
│   ├── batch_processing.py
│   └── data_conversion.py
├── data/                        # Sample data (keep structure)
│   └── ...
├── outputs/                     # Output files (keep structure)
│   └── ...
├── results/                     # Results (keep structure)
│   └── ...
└── experimental_tracker/        # Experimental code (keep structure)
    └── ...
```

## Key Improvements

### 1. **Modular Architecture**

- Separated core business logic from UI
- Clear module boundaries with `__init__.py` files
- Grouped related functionality together

### 2. **Domain-Driven Design**

- **Core Package**: Contains the main business logic
  - `detection/`: Fish detection algorithms
  - `tracking/`: Object tracking functionality
  - `fish/`: Fish-specific data management
  - `processing/`: Data processing workflows

### 3. **UI Separation**

- All UI components in dedicated `ui/` package
- Widgets grouped by functionality
- Dialogs and views separated for clarity

### 4. **Configuration Management**

- Centralized configuration files
- Parameter classes organized by domain
- Easy to extend and maintain

### 5. **Testing Structure**

- Mirror the main package structure in tests
- Separate test files for each module
- Proper test configuration with `conftest.py`

### 6. **Documentation**

- Comprehensive documentation structure
- API documentation
- User and developer guides

## Migration Steps

### Phase 1: Create New Structure

1. Create the new directory structure
2. Move files to appropriate locations
3. Update import statements

### Phase 2: Refactor Imports

1. Update all import statements to use the new structure
2. Add proper `__init__.py` files
3. Test that all imports work correctly

### Phase 3: Update Configuration

1. Update `pyproject.toml` for the new package structure
2. Create proper entry points
3. Update any build scripts

### Phase 4: Testing and Documentation

1. Set up proper testing framework
2. Update documentation
3. Add CI/CD configuration

## Benefits of This Structure

1. **Maintainability**: Clear separation makes code easier to maintain
2. **Scalability**: Easy to add new features without cluttering
3. **Testability**: Proper structure supports comprehensive testing
4. **Collaboration**: Team members can work on different modules independently
5. **Distribution**: Package can be easily distributed and installed
6. **Professional**: Follows Python packaging best practices

## Implementation Priority

1. **High Priority**: Core logic separation (detection, tracking, fish management)
2. **Medium Priority**: UI restructuring and parameter management
3. **Low Priority**: Documentation and testing structure (can be done incrementally)

This structure follows industry best practices seen in projects like Ultralytics YOLO and OpenCV, providing a solid foundation for your fish tracking application.
