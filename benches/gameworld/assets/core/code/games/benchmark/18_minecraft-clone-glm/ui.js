/**
 * UI System
 * HUD, menus, debug info
 */

// UI state
let uiState = {
    isInventoryOpen: false,
    isPauseMenuOpen: false,
    isDebugVisible: false,
    isLoading: true
};

// Initialize UI
function initUI() {
    // Setup main menu buttons
    document.getElementById('btn-new-world').addEventListener('click', () => {
        startNewGame();
    });

    document.getElementById('btn-load-world').addEventListener('click', () => {
        loadGame();
    });

    document.getElementById('btn-resume').addEventListener('click', () => {
        resumeGame();
    });

    document.getElementById('btn-save').addEventListener('click', () => {
        saveGame();
        showNotification('World Saved!');
    });

    document.getElementById('btn-quit').addEventListener('click', () => {
        quitToMenu();
    });

    // Setup hotbar scroll
    document.addEventListener('wheel', (e) => {
        if (!player || !player.isLocked) return;
        scrollHotbar(e.deltaY > 0 ? 1 : -1);
    });

    // Setup keyboard for UI
    document.addEventListener('keydown', (e) => {
        handleUIKey(e);
    });

    // Initialize health and hunger bars
    updateHealthDisplay();
    updateHungerDisplay();

    // Initialize inventory UI
    initInventoryUI();
}

// Show loading screen
function showLoading(message) {
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('loading').querySelector('div').textContent = message || 'Loading...';
}

// Update loading progress
function updateLoadingProgress(progress) {
    document.getElementById('loading-progress').style.width = `${progress * 100}%`;
}

// Hide loading screen
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
    uiState.isLoading = false;
}

// Show main menu
function showMainMenu() {
    document.getElementById('main-menu').style.display = 'flex';
    document.getElementById('pause-menu').style.display = 'none';
    document.getElementById('inventory-screen').style.display = 'none';
    uiState.isPauseMenuOpen = false;
    uiState.isInventoryOpen = false;
}

// Hide main menu
function hideMainMenu() {
    document.getElementById('main-menu').style.display = 'none';
}

// Show pause menu
function showPauseMenu() {
    document.getElementById('pause-menu').style.display = 'flex';
    uiState.isPauseMenuOpen = true;
    if (game) game.pause();
}

// Hide pause menu (does NOT resume game or request pointer lock — use resumeGame() for that)
function hidePauseMenu() {
    document.getElementById('pause-menu').style.display = 'none';
    uiState.isPauseMenuOpen = false;
}

// Toggle pause menu
function togglePauseMenu() {
    if (uiState.isPauseMenuOpen) {
        hidePauseMenu();
        if (player) player.requestPointerLock();
    } else {
        showPauseMenu();
        if (player) player.exitPointerLock();
    }
}

// Update health display
function updateHealthDisplay() {
    const healthBar = document.getElementById('health-bar');
    healthBar.innerHTML = '';

    const hearts = Math.ceil((player ? player.health : 20) / 2);

    for (let i = 0; i < 10; i++) {
        const heart = document.createElement('div');
        heart.className = 'heart';

        // Create heart icon with canvas
        const canvas = document.createElement('canvas');
        canvas.width = 16;
        canvas.height = 16;
        const ctx = canvas.getContext('2d');

        if (i < hearts) {
            // Full heart
            ctx.fillStyle = '#ff0000';
            drawHeart(ctx, 8, 8, 6);
        } else {
            // Empty heart
            ctx.fillStyle = '#400000';
            drawHeart(ctx, 8, 8, 6);
            ctx.fillStyle = '#800000';
            ctx.strokeStyle = '#800000';
            ctx.lineWidth = 1;
            drawHeartOutline(ctx, 8, 8, 6);
        }

        heart.style.backgroundImage = `url(${canvas.toDataURL()})`;
        healthBar.appendChild(heart);
    }
}

// Draw heart shape
function drawHeart(ctx, x, y, size) {
    ctx.beginPath();
    ctx.moveTo(x, y - size * 0.4);
    ctx.bezierCurveTo(x + size, y - size * 1.2, x + size, y + size * 0.2, x, y + size * 0.6);
    ctx.bezierCurveTo(x - size, y + size * 0.2, x - size, y - size * 1.2, x, y - size * 0.4);
    ctx.fill();
}

// Draw heart outline
function drawHeartOutline(ctx, x, y, size) {
    ctx.beginPath();
    ctx.moveTo(x, y - size * 0.4);
    ctx.bezierCurveTo(x + size, y - size * 1.2, x + size, y + size * 0.2, x, y + size * 0.6);
    ctx.bezierCurveTo(x - size, y + size * 0.2, x - size, y - size * 1.2, x, y - size * 0.4);
    ctx.stroke();
}

// Update hunger display
function updateHungerDisplay() {
    const hungerBar = document.getElementById('hunger-bar');
    hungerBar.innerHTML = '';

    const food = Math.ceil((player ? player.hunger : 20) / 2);

    for (let i = 0; i < 10; i++) {
        const hunger = document.createElement('div');
        hunger.className = 'hunger';

        // Create food icon with canvas
        const canvas = document.createElement('canvas');
        canvas.width = 16;
        canvas.height = 16;
        const ctx = canvas.getContext('2d');

        if (i < food) {
            // Full food
            ctx.fillStyle = '#8B4513';
            ctx.beginPath();
            ctx.arc(8, 8, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#CD853F';
            ctx.beginPath();
            ctx.arc(8, 8, 3, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // Empty food
            ctx.fillStyle = '#3a2a1a';
            ctx.beginPath();
            ctx.arc(8, 8, 5, 0, Math.PI * 2);
            ctx.fill();
        }

        hunger.style.backgroundImage = `url(${canvas.toDataURL()})`;
        hungerBar.appendChild(hunger);
    }
}

// Update debug info
function updateDebugInfo() {
    if (!uiState.isDebugVisible) return;

    const debugInfo = document.getElementById('debug-info');
    if (!player || !chunkManager) {
        debugInfo.innerHTML = 'Loading...';
        return;
    }

    const pos = player.position;
    const chunk = chunkManager.getChunkAtWorld(pos.x, pos.z);
    const fps = game ? game.fps.toFixed(0) : 0;
    const chunks = chunkManager.chunks.size;
    const time = getTimeString();

    debugInfo.innerHTML = `
        Minecraft Clone v1.0<br>
        <br>
        XYZ: ${pos.x.toFixed(2)} / ${pos.y.toFixed(2)} / ${pos.z.toFixed(2)}<br>
        Chunk: ${chunk ? `${chunk.cx}, ${chunk.cz}` : 'N/A'}<br>
        Loaded Chunks: ${chunks}<br>
        <br>
        Time: ${time}<br>
        Day: ${lightingState.dayCount + 1}<br>
        <br>
        FPS: ${fps}<br>
        ${player.isInWater ? 'In Water<br>' : ''}
        ${player.isSprinting ? 'Sprinting<br>' : ''}
    `;
}

// Toggle debug info
function toggleDebugInfo() {
    uiState.isDebugVisible = !uiState.isDebugVisible;
    document.getElementById('debug-info').style.display = uiState.isDebugVisible ? 'block' : 'none';
}

// Handle UI keyboard input
function handleUIKey(e) {
    // Escape key
    // NOTE: When pointer lock is active, the browser intercepts ESC to exit
    // pointer lock, and pointerlockchange handler will open the pause menu.
    // So here we ONLY handle closing UI overlays — never opening the pause menu.
    if (e.code === 'Escape') {
        if (uiState.isInventoryOpen) {
            toggleInventory();
        } else if (uiState.isPauseMenuOpen) {
            // Resume game from pause
            resumeGame();
        }
        return;
    }

    // F3 - debug info
    if (e.code === 'F3') {
        e.preventDefault();
        toggleDebugInfo();
        return;
    }

    // E - inventory (works both to open AND close)
    if (e.code === 'KeyE') {
        if (uiState.isInventoryOpen) {
            toggleInventory();
            return;
        } else if (player && player.isLocked) {
            toggleInventory();
            return;
        }
    }

    // Number keys for hotbar (only when playing)
    if (e.code >= 'Digit1' && e.code <= 'Digit9' && player && player.isLocked) {
        const slot = parseInt(e.code.replace('Digit', '')) - 1;
        selectHotbarSlot(slot);
    }
}

// Update underwater overlay
function updateUnderwaterOverlay() {
    const overlay = document.getElementById('underwater-overlay');
    if (player && player.isUnderwater) {
        overlay.style.display = 'block';
    } else {
        overlay.style.display = 'none';
    }
}

// Show notification
function showNotification(message, duration = 2000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 15px 30px;
        border-radius: 5px;
        font-size: 18px;
        z-index: 2000;
        pointer-events: none;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, duration);
}

// Show crosshair
function showCrosshair() {
    document.getElementById('crosshair').style.display = 'block';
}

// Hide crosshair
function hideCrosshair() {
    document.getElementById('crosshair').style.display = 'none';
}

// Show click to play overlay
function showClickToPlay() {
    document.getElementById('click-to-play').style.display = 'block';
}

// Hide click to play overlay
function hideClickToPlay() {
    document.getElementById('click-to-play').style.display = 'none';
}

// Update UI every frame
function updateUI() {
    updateUnderwaterOverlay();

    if (uiState.isDebugVisible) {
        updateDebugInfo();
    }
}

// Check if any UI is open
function isUIOpen() {
    return uiState.isInventoryOpen || uiState.isPauseMenuOpen || uiState.isLoading;
}
