# PTOLEMY UI

A celestial-themed user interface for the PTOLEMY project, built with FastAPI and Jinja2.

## Overview

The PTOLEMY UI provides a beautiful, celestial-themed interface for interacting with the PTOLEMY content generation system. It features:

- Project management dashboard with celestial visualization
- Content generation interface
- Feedback analysis and timeline visualization
- Responsive design with a space-themed aesthetic

## Getting Started

### Prerequisites

- Python 3.8+
- PTOLEMY core installed

### Running the UI

From the project root directory, run:

```bash
python run_ui.py
```

This will start the FastAPI server at http://127.0.0.1:8000

## UI Structure

### Routes

- `/` - Home page
- `/projects` - Project management
- `/projects/{project_id}` - Project details
- `/generate` - Content generation
- `/analysis` - Feedback analysis
- `/analysis/timeline` - Event timeline

### Templates

- `base.html` - Base template with common layout elements
- `index.html` - Landing page
- `projects.html` - Project listing with celestial sphere visualization
- `project_details.html` - Detailed project view
- `generation.html` - Content generation interface
- `analysis.html` - Feedback analysis dashboard
- `timeline.html` - Timeline visualization

### Static Assets

- `css/celestial.css` - Celestial theme styles
- `js/celestial.js` - Interactive visualizations and UI functionality

## Features

### Celestial Sphere Project Visualization

Projects are visualized as celestial bodies orbiting in a celestial sphere. The size and position of each project body represents its importance and relationship to other projects.

### Content Generation

The generation interface provides a simple form for creating content using the PTOLEMY system, with options to select AI models and customize parameters.

### Analysis Dashboard

The analysis dashboard provides insights into feedback and system performance, with visualizations of sentiment analysis and trends.

### Timeline Visualization

The timeline view shows a chronological visualization of events related to projects, with celestial-themed styling for event markers.

## Customization

The UI can be customized by modifying the CSS and JavaScript files in the `static` directory. The celestial theme can be adjusted by changing the color variables in `celestial.css`.

## Integration with PTOLEMY Core

The UI integrates with the PTOLEMY core functionality through the route handlers, which call the appropriate methods from the PTOLEMY API.
