/**
 * Lighting and Sky System
 * Day/night cycle, sky gradient, fog, stars
 */

// Day length in real seconds
const DAY_LENGTH = 1200; // 20 minutes

// Lighting state
let lightingState = {
    time: 0.25, // Start at morning (0-1, 0.25 = 6AM)
    dayCount: 0,
    sunAngle: 0,
    skyColor: new THREE.Color(0x87CEEB),
    fogColor: new THREE.Color(0x87CEEB),
    ambientIntensity: 0.4,
    sunIntensity: 1.0,
    moonIntensity: 0.2
};

// Sky colors for different times
const SKY_COLORS = {
    dawn: { top: 0xFFB347, bottom: 0xFF6B6B },
    day: { top: 0x87CEEB, bottom: 0xB0E0E6 },
    dusk: { top: 0xFF7F50, bottom: 0xDC143C },
    night: { top: 0x0a0a20, bottom: 0x1a1a3a }
};

// Sky dome
let skyDome = null;
let sunMesh = null;
let moonMesh = null;
let starsMesh = null;

// Create sky dome
function createSkyDome(scene) {
    // Sky dome geometry
    const skyGeo = new THREE.SphereGeometry(500, 32, 32);
    const skyMat = new THREE.ShaderMaterial({
        uniforms: {
            topColor: { value: new THREE.Color(SKY_COLORS.day.top) },
            bottomColor: { value: new THREE.Color(SKY_COLORS.day.bottom) },
            offset: { value: 20 },
            exponent: { value: 0.6 }
        },
        vertexShader: `
            varying vec3 vWorldPosition;
            void main() {
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 topColor;
            uniform vec3 bottomColor;
            uniform float offset;
            uniform float exponent;
            varying vec3 vWorldPosition;
            void main() {
                float h = normalize(vWorldPosition + offset).y;
                gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
            }
        `,
        side: THREE.BackSide
    });

    skyDome = new THREE.Mesh(skyGeo, skyMat);
    scene.add(skyDome);

    // Create sun
    const sunGeo = new THREE.CircleGeometry(20, 32);
    const sunMat = new THREE.MeshBasicMaterial({
        color: 0xFFFF00,
        transparent: true,
        opacity: 0.9
    });
    sunMesh = new THREE.Mesh(sunGeo, sunMat);
    scene.add(sunMesh);

    // Create moon
    const moonGeo = new THREE.CircleGeometry(15, 32);
    const moonMat = new THREE.MeshBasicMaterial({
        color: 0xEEEEEE,
        transparent: true,
        opacity: 0.8
    });
    moonMesh = new THREE.Mesh(moonGeo, moonMat);
    scene.add(moonMesh);

    // Create stars
    createStars(scene);

    return skyDome;
}

// Create star field
function createStars(scene) {
    const starCount = 1000;
    const positions = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);

    for (let i = 0; i < starCount; i++) {
        // Random position on sphere
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 400;

        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.cos(phi);
        positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);

        sizes[i] = Math.random() * 2 + 0.5;
    }

    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starGeo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const starMat = new THREE.PointsMaterial({
        color: 0xFFFFFF,
        size: 2,
        transparent: true,
        opacity: 0,
        sizeAttenuation: false
    });

    starsMesh = new THREE.Points(starGeo, starMat);
    scene.add(starsMesh);
}

// Update lighting based on time
function updateLighting(scene, deltaTime, camera) {
    // Update time
    lightingState.time += deltaTime / DAY_LENGTH;
    if (lightingState.time >= 1) {
        lightingState.time -= 1;
        lightingState.dayCount++;
    }

    const time = lightingState.time;
    const sunAngle = time * Math.PI * 2 - Math.PI / 2;

    // Calculate sun/moon position
    const skyRadius = 300;

    // Sun position
    sunMesh.position.set(
        camera.position.x + Math.cos(sunAngle) * skyRadius,
        camera.position.y + Math.sin(sunAngle) * skyRadius,
        camera.position.z
    );
    sunMesh.lookAt(camera.position);

    // Moon position (opposite to sun)
    moonMesh.position.set(
        camera.position.x + Math.cos(sunAngle + Math.PI) * skyRadius,
        camera.position.y + Math.sin(sunAngle + Math.PI) * skyRadius,
        camera.position.z
    );
    moonMesh.lookAt(camera.position);

    // Update sky dome position to follow camera
    skyDome.position.copy(camera.position);

    // Update stars position
    if (starsMesh) {
        starsMesh.position.copy(camera.position);
    }

    // Determine sky colors based on time
    let topColor, bottomColor, ambientIntensity, sunIntensity, starOpacity;

    if (time < 0.2) {
        // Night (0 - 0.2)
        const t = time / 0.2;
        topColor = lerpColor(SKY_COLORS.night.top, SKY_COLORS.dawn.top, t);
        bottomColor = lerpColor(SKY_COLORS.night.bottom, SKY_COLORS.dawn.bottom, t);
        ambientIntensity = 0.3;
        sunIntensity = 0;
        starOpacity = 1 - t;
    } else if (time < 0.3) {
        // Dawn (0.2 - 0.3)
        const t = (time - 0.2) / 0.1;
        topColor = lerpColor(SKY_COLORS.dawn.top, SKY_COLORS.day.top, t);
        bottomColor = lerpColor(SKY_COLORS.dawn.bottom, SKY_COLORS.day.bottom, t);
        ambientIntensity = 0.3 + t * 0.3;
        sunIntensity = t;
        starOpacity = 0;
    } else if (time < 0.7) {
        // Day (0.3 - 0.7)
        topColor = SKY_COLORS.day.top;
        bottomColor = SKY_COLORS.day.bottom;
        ambientIntensity = 0.6;
        sunIntensity = 1;
        starOpacity = 0;
    } else if (time < 0.8) {
        // Dusk (0.7 - 0.8)
        const t = (time - 0.7) / 0.1;
        topColor = lerpColor(SKY_COLORS.day.top, SKY_COLORS.dusk.top, t);
        bottomColor = lerpColor(SKY_COLORS.day.bottom, SKY_COLORS.dusk.bottom, t);
        ambientIntensity = 0.6 - t * 0.3;
        sunIntensity = 1 - t;
        starOpacity = t;
    } else {
        // Night (0.8 - 1.0)
        const t = (time - 0.8) / 0.2;
        topColor = lerpColor(SKY_COLORS.dusk.top, SKY_COLORS.night.top, t);
        bottomColor = lerpColor(SKY_COLORS.dusk.bottom, SKY_COLORS.night.bottom, t);
        ambientIntensity = 0.3;
        sunIntensity = 0;
        starOpacity = 1;
    }

    // Update sky dome shader
    skyDome.material.uniforms.topColor.value.setHex(topColor);
    skyDome.material.uniforms.bottomColor.value.setHex(bottomColor);

    // Update fog color
    scene.fog.color.setHex(bottomColor);

    // Update lights
    const ambientLight = scene.getObjectByName('ambientLight');
    const directionalLight = scene.getObjectByName('sunLight');

    if (ambientLight) {
        ambientLight.intensity = ambientIntensity;
    }

    if (directionalLight) {
        directionalLight.intensity = sunIntensity;
        directionalLight.position.copy(sunMesh.position).sub(camera.position).normalize();
    }

    // Update stars visibility
    if (starsMesh) {
        starsMesh.material.opacity = starOpacity;
    }

    // Update sun/moon visibility
    sunMesh.visible = sunIntensity > 0;
    sunMesh.material.opacity = sunIntensity * 0.9;

    moonMesh.visible = sunIntensity < 0.5;
    moonMesh.material.opacity = (1 - sunIntensity) * 0.8;

    // Store state
    lightingState.skyColor.setHex(topColor);
    lightingState.fogColor.setHex(bottomColor);
    lightingState.ambientIntensity = ambientIntensity;
    lightingState.sunIntensity = sunIntensity;
    lightingState.sunAngle = sunAngle;
}

// Lerp between two hex colors
function lerpColor(color1, color2, t) {
    const c1 = new THREE.Color(color1);
    const c2 = new THREE.Color(color2);
    c1.lerp(c2, t);
    return c1.getHex();
}

// Get current time of day (0-1)
function getDayTime() {
    return lightingState.time;
}

// Get formatted time string
function getTimeString() {
    const hours = Math.floor(lightingState.time * 24);
    const minutes = Math.floor((lightingState.time * 24 * 60) % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
}

// Check if it's night time
function isNight() {
    return lightingState.time < 0.25 || lightingState.time > 0.75;
}

// Get sun position for shadows
function getSunDirection() {
    const angle = lightingState.sunAngle;
    return new THREE.Vector3(Math.cos(angle), Math.sin(angle), 0.5).normalize();
}

// Setup scene fog
function setupFog(scene) {
    scene.fog = new THREE.Fog(SKY_COLORS.day.bottom, 50, 300);
}

// Setup scene lights
function setupLighting(scene) {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    ambientLight.name = 'ambientLight';
    scene.add(ambientLight);

    // Directional light (sun)
    const sunLight = new THREE.DirectionalLight(0xffffff, 1);
    sunLight.name = 'sunLight';
    sunLight.position.set(100, 200, 50);
    scene.add(sunLight);

    // Hemisphere light for better ambient
    const hemiLight = new THREE.HemisphereLight(0x87CEEB, 0x8B4513, 0.3);
    scene.add(hemiLight);
}

// Get lighting state for saving
function getLightingState() {
    return {
        time: lightingState.time,
        dayCount: lightingState.dayCount
    };
}

// Load lighting state from save
function loadLightingState(state) {
    if (state.time !== undefined) lightingState.time = state.time;
    if (state.dayCount !== undefined) lightingState.dayCount = state.dayCount;
}
