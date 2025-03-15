/**
 * PTOLEMY UI - Celestial JavaScript
 * Handles interactive elements and visualizations for the PTOLEMY UI
 */

// Project Celestial Sphere Visualization
function initCelestialSphere(projects) {
    const sphere = document.getElementById('celestial-sphere');
    if (!sphere || !projects) return;
    
    const centerX = sphere.offsetWidth / 2;
    const centerY = sphere.offsetHeight / 2;
    
    // Clear existing project bodies
    while (sphere.firstChild) {
        sphere.removeChild(sphere.firstChild);
    }
    
    // Create orbits
    const orbits = 3;
    for (let i = 1; i <= orbits; i++) {
        const orbit = document.createElement('div');
        orbit.className = 'orbit';
        const size = (sphere.offsetWidth * i) / (orbits + 1);
        orbit.style.width = `${size}px`;
        orbit.style.height = `${size}px`;
        sphere.appendChild(orbit);
    }
    
    // Add project bodies
    projects.forEach((project, index) => {
        // Calculate position on a spiral
        const angle = (index / projects.length) * Math.PI * 4; // Spiral around twice
        const distance = 50 + (index / projects.length) * (sphere.offsetWidth / 2 - 100);
        const x = centerX + distance * Math.cos(angle);
        const y = centerY + distance * Math.sin(angle);
        
        // Size based on project importance or activity
        const size = 30 + (project.activity_level || 1) * 10;
        
        // Create project body
        const body = document.createElement('div');
        body.className = 'project-body';
        body.style.width = `${size}px`;
        body.style.height = `${size}px`;
        body.style.left = `${x - size/2}px`;
        body.style.top = `${y - size/2}px`;
        body.textContent = project.name.charAt(0).toUpperCase();
        body.title = project.name;
        body.dataset.projectId = project.id;
        
        // Add click handler
        body.addEventListener('click', () => {
            window.location.href = `/projects/${project.id}`;
        });
        
        sphere.appendChild(body);
    });
}

// Timeline Visualization
function initTimeline() {
    const timeline = document.getElementById('timeline-container');
    if (!timeline) return;
    
    // Add animation to timeline events
    const events = timeline.querySelectorAll('.timeline-event');
    events.forEach((event, index) => {
        event.style.opacity = '0';
        event.style.transform = 'translateX(-20px)';
        event.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        
        // Stagger the animations
        setTimeout(() => {
            event.style.opacity = '1';
            event.style.transform = 'translateX(0)';
        }, 100 * index);
    });
}

// Celestial Loader
function showLoader(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const loader = document.createElement('div');
    loader.className = 'celestial-loader';
    loader.innerHTML = `
        <div class="orbit"></div>
        <div class="planet"></div>
    `;
    
    container.appendChild(loader);
    return loader;
}

function hideLoader(loader) {
    if (loader && loader.parentNode) {
        loader.parentNode.removeChild(loader);
    }
}

// Content Generation
function initGenerationForm() {
    const form = document.getElementById('generation-form');
    const resultContainer = document.getElementById('generation-result');
    
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        const loader = showLoader('generation-container');
        
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Generation failed');
            }
            
            const result = await response.json();
            
            if (resultContainer) {
                resultContainer.innerHTML = `
                    <div class="bg-deep-space/50 backdrop-blur-sm p-4 rounded-lg border border-celestial-blue/30 mt-4">
                        <h3 class="text-xl font-bold mb-2 text-celestial-blue">Generated Content</h3>
                        <div class="whitespace-pre-wrap">${result.content}</div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error:', error);
            if (resultContainer) {
                resultContainer.innerHTML = `
                    <div class="bg-deep-space/50 backdrop-blur-sm p-4 rounded-lg border border-red-500/30 mt-4">
                        <h3 class="text-xl font-bold mb-2 text-red-500">Error</h3>
                        <p>Failed to generate content. Please try again.</p>
                    </div>
                `;
            }
        } finally {
            hideLoader(loader);
        }
    });
}

// Initialize all components when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get projects data if available
    const projectsData = window.projectsData || [];
    
    // Initialize visualizations
    initCelestialSphere(projectsData);
    initTimeline();
    initGenerationForm();
    
    // Add stars to background
    addStarsToBackground();
});

// Add animated stars to the background
function addStarsToBackground() {
    const body = document.body;
    const starsContainer = document.createElement('div');
    starsContainer.className = 'stars-container';
    starsContainer.style.position = 'fixed';
    starsContainer.style.top = '0';
    starsContainer.style.left = '0';
    starsContainer.style.width = '100%';
    starsContainer.style.height = '100%';
    starsContainer.style.pointerEvents = 'none';
    starsContainer.style.zIndex = '-1';
    
    // Add stars
    for (let i = 0; i < 100; i++) {
        const star = document.createElement('div');
        const size = Math.random() * 2 + 1;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const duration = Math.random() * 3 + 2;
        const delay = Math.random() * 5;
        
        star.style.position = 'absolute';
        star.style.width = `${size}px`;
        star.style.height = `${size}px`;
        star.style.borderRadius = '50%';
        star.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        star.style.left = `${x}%`;
        star.style.top = `${y}%`;
        star.style.animation = `twinkle ${duration}s infinite ${delay}s`;
        
        starsContainer.appendChild(star);
    }
    
    // Add keyframes for twinkling animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes twinkle {
            0%, 100% { opacity: 0.2; }
            50% { opacity: 1; }
        }
    `;
    
    document.head.appendChild(style);
    body.appendChild(starsContainer);
}
