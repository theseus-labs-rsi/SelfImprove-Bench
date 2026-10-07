/**
 * Main Game Entry Point
 * Game loop, initialization, and state management
 */

// Game state
let game = null;
let worldSeed = Date.now();

// Game class
class Game {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = null;

        this.isRunning = false;
        this.isPaused = false;
        this.fps = 0;
        this.frameCount = 0;
        this.fpsTime = 0;

        this.lastTime = 0;
    }

    init() {
        // Create texture atlas FIRST (before any chunks are built)
        createTextureAtlas();

        // Create scene
        this.scene = new THREE.Scene();

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            70,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );

        // Add camera to scene so that children (hand model) are rendered
        this.scene.add(this.camera);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: false,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        document.getElementById('game-container').appendChild(this.renderer.domElement);

        // Clock for delta time
        this.clock = new THREE.Clock();

        // Setup fog
        setupFog(this.scene);

        // Setup lighting
        setupLighting(this.scene);

        // Initialize systems
        initAudio();
        initChunkManager(this.scene);
        initPlayer(this.camera);
        initInventory();
        initMobManager(this.scene);
        initParticles(this.scene);
        initBlockInteraction();
        // initUI() already called on page load

        // Add block highlight to scene
        const highlight = createBlockHighlight();
        this.scene.add(highlight);

        // Create sky dome
        createSkyDome(this.scene);

        // Handle window resize
        window.addEventListener('resize', () => this.onResize());

        // Start render loop
        this.animate();

        console.log('Game initialized');
    }

    start(seed, saveData = null) {
        worldSeed = seed || Date.now();

        // Initialize noise with seed
        initNoise(worldSeed);

        // Generate initial chunks
        this.generateWorld(saveData);

        // Apply save data if exists
        if (saveData) {
            applySaveData(saveData);
        }

        // Start auto-save
        startAutoSave(60000);

        this.isRunning = true;
        hideMainMenu();
        hideLoading();
        showCrosshair();
        showClickToPlay();

        console.log('Game started with seed:', worldSeed);
    }

    generateWorld(saveData) {
        showLoading('Generating world...');
        updateLoadingProgress(0);

        // Set player spawn position
        const spawn = { x: 8, y: 80, z: 8 };
        player.position.set(spawn.x, spawn.y, spawn.z);

        // Generate initial chunks
        const totalChunks = (RENDER_DISTANCE * 2 + 1) ** 2;
        let generatedChunks = 0;

        const pcx = Math.floor(spawn.x / CHUNK_SIZE);
        const pcz = Math.floor(spawn.z / CHUNK_SIZE);

        for (let dx = -RENDER_DISTANCE; dx <= RENDER_DISTANCE; dx++) {
            for (let dz = -RENDER_DISTANCE; dz <= RENDER_DISTANCE; dz++) {
                const cx = pcx + dx;
                const cz = pcz + dz;

                const chunk = chunkManager.createChunk(cx, cz);
                generateTerrain(chunkManager, cx, cz);
                chunk.isGenerated = true;

                generatedChunks++;
                updateLoadingProgress(generatedChunks / totalChunks);
            }
        }

        // Build initial meshes
        for (const chunk of chunkManager.chunks.values()) {
            chunkManager.buildChunkMesh(chunk);
        }

        // Find actual spawn position
        const actualSpawn = getSpawnPosition(chunkManager);
        player.position.set(actualSpawn.x, actualSpawn.y, actualSpawn.z);

        // Update camera immediately so the world is visible before pointer lock
        player.updateCamera();

        updateLoadingProgress(1);
    }

    pause() {
        this.isPaused = true;
    }

    resume() {
        this.isPaused = false;
        this.clock.getDelta(); // Reset delta
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const deltaTime = Math.min(this.clock.getDelta(), 0.1);

        // Calculate FPS
        this.frameCount++;
        this.fpsTime += deltaTime;
        if (this.fpsTime >= 1) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.fpsTime = 0;
        }

        // Update game if running and not paused
        if (this.isRunning && !this.isPaused) {
            // Always update lighting and chunks even without pointer lock
            updateLighting(this.scene, deltaTime, this.camera);

            if (player && player.isLocked) {
                this.update(deltaTime);
            } else if (player) {
                // Keep camera synced to player position even without pointer lock
                player.updateCamera();
            }
        }

        // Always render
        this.render();
    }

    update(deltaTime) {
        // Update player
        player.update(deltaTime);

        // Update chunks around player
        chunkManager.updateChunksAroundPlayer(
            player.position.x,
            player.position.z
        );

        // Update block interaction
        updateBlockInteraction(deltaTime);

        // Note: updateLighting is called in animate() to avoid double-calling

        // Update mobs
        mobManager.update(deltaTime, player, getDayTime());

        // Update particles
        updateParticles(deltaTime);

        // Update UI
        updateUI();
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }

    dispose() {
        this.isRunning = false;
        stopAutoSave();

        // Clear all systems
        if (chunkManager) {
            for (const chunk of chunkManager.chunks.values()) {
                if (chunk.mesh) {
                    chunk.mesh.geometry.dispose();
                    this.scene.remove(chunk.mesh);
                }
                if (chunk.waterMesh) {
                    chunk.waterMesh.geometry.dispose();
                    this.scene.remove(chunk.waterMesh);
                }
                if (chunk.spriteMesh) {
                    chunk.spriteMesh.geometry.dispose();
                    this.scene.remove(chunk.spriteMesh);
                }
            }
            chunkManager.chunks.clear();
        }

        if (mobManager) {
            mobManager.clear();
        }

        clearParticles();

        this.renderer.dispose();
    }
}

// Start new game
function startNewGame() {
    showLoading('Creating new world...');

    if (game) {
        game.dispose();
    }

    game = new Game();
    game.init();
    game.start(Date.now());
}

// Start game with seed
function startGame(seed, saveData = null) {
    showLoading('Loading world...');

    if (game) {
        game.dispose();
    }

    game = new Game();
    game.init();
    game.start(seed, saveData);
}

// Resume game from pause
function resumeGame() {
    hidePauseMenu();
    if (game) game.resume();
    if (player) player.requestPointerLock();
}

// Quit to main menu
function quitToMenu() {
    if (game) {
        saveGame();
        game.dispose();
        game = null;
    }

    showMainMenu();
    hidePauseMenu();
    hideCrosshair();
}

// Initialize on load
window.addEventListener('load', () => {
    // Initialize audio on first interaction
    document.addEventListener('click', () => {
        resumeAudio();
    }, { once: true });

    // Initialize UI immediately (for menu buttons)
    initUI();

    // Show main menu
    showMainMenu();
    hideLoading();

    // Check for existing save
    if (hasSaveData()) {
        document.getElementById('btn-load-world').style.opacity = '1';
    } else {
        document.getElementById('btn-load-world').style.opacity = '0.5';
    }
});

// Prevent context menu on right click
document.addEventListener('contextmenu', (e) => e.preventDefault());

// Handle visibility change (pause when tab hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden && game && game.isRunning && !game.isPaused) {
        showPauseMenu();
    }
});

// Handle beforeunload (save on close)
window.addEventListener('beforeunload', () => {
    if (game && game.isRunning) {
        saveGame();
    }
});

// Console commands for debugging
window.mc = {
    save: saveGame,
    load: loadGame,
    setDay: () => { lightingState.time = 0.4; },
    setNight: () => { lightingState.time = 0; },
    give: (blockType, count = 64) => addToInventory(blockType, count),
    fly: () => { if (player) player.velocity.y = 10; },
    spawnMob: (type = 'zombie') => {
        if (mobManager && player) {
            const mob = new Mob(type, player.position.x + 5, player.position.y, player.position.z);
            mobManager.mobs.push(mob);
            mobManager.scene.add(mob.mesh);
        }
    },
    seed: () => worldSeed,
    pos: () => player ? player.position : null
};
