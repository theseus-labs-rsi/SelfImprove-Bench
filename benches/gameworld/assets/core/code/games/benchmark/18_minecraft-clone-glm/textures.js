/**
 * Procedural Texture Generation
 * All textures are 16x16, generated with Canvas
 */

const TEXTURE_SIZE = 16;

// Block types
const BlockType = {
    AIR: 0,
    GRASS: 1,
    DIRT: 2,
    STONE: 3,
    WOOD_LOG: 4,
    LEAVES: 5,
    SAND: 6,
    WATER: 7,
    BEDROCK: 8,
    WOOD_PLANK: 9,
    COBBLESTONE: 10,
    COAL_ORE: 11,
    IRON_ORE: 12,
    GOLD_ORE: 13,
    DIAMOND_ORE: 14,
    TORCH: 15,
    GLASS: 16,
    CRAFTING_TABLE: 17,
    SNOW: 18,
    CACTUS: 19,
    // Tool/item types (not placeable)
    STICK: 20,
    WOODEN_PICKAXE: 21,
    WOODEN_AXE: 22,
    WOODEN_SWORD: 23,
    WOODEN_SHOVEL: 24,
    WOODEN_HOE: 25
};

// Texture atlas
let textureAtlas = null;
let textureCanvas = null;
let textureUVs = {};

// Seeded random for consistent textures
function textureRandom(seed) {
    const m = 0x80000000;
    const a = 1103515245;
    const c = 12345;
    let state = seed;
    return function() {
        state = (a * state + c) % m;
        return state / m;
    };
}

// Create a 16x16 texture canvas
function createTextureCanvas() {
    const canvas = document.createElement('canvas');
    canvas.width = TEXTURE_SIZE;
    canvas.height = TEXTURE_SIZE;
    return canvas;
}

// Add noise to texture
function addNoise(ctx, rand, intensity = 0.1) {
    const imageData = ctx.getImageData(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        const noise = (rand() - 0.5) * intensity * 255;
        data[i] = Math.max(0, Math.min(255, data[i] + noise));
        data[i+1] = Math.max(0, Math.min(255, data[i+1] + noise));
        data[i+2] = Math.max(0, Math.min(255, data[i+2] + noise));
    }
    ctx.putImageData(imageData, 0, 0);
}

// Grass top texture
function generateGrassTop() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(1);

    ctx.fillStyle = '#5d9b3d';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Add grass variation
    for (let i = 0; i < 40; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 30) - 15;
        ctx.fillStyle = `rgb(${93+shade}, ${155+shade}, ${61+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Grass side texture
function generateGrassSide() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(2);

    // Dirt base
    ctx.fillStyle = '#8b6b4a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Add dirt variation
    for (let i = 0; i < 30; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 40) - 20;
        ctx.fillStyle = `rgb(${139+shade}, ${107+shade}, ${74+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    // Grass top strip — only 1px at the very top
    ctx.fillStyle = '#5d9b3d';
    ctx.fillRect(0, 0, TEXTURE_SIZE, 1);

    // Sparse grass overhang — just a few irregular pixels hanging down
    for (let x = 0; x < TEXTURE_SIZE; x++) {
        if (rand() > 0.5) {
            ctx.fillStyle = `rgb(${70+Math.floor(rand()*20)}, ${130+Math.floor(rand()*20)}, ${45+Math.floor(rand()*15)})`;
            ctx.fillRect(x, 1, 1, 1);
        }
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Dirt texture
function generateDirt() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(3);

    ctx.fillStyle = '#8b6b4a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    for (let i = 0; i < 50; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 50) - 25;
        ctx.fillStyle = `rgb(${139+shade}, ${107+shade}, ${74+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.08);
    return canvas;
}

// Stone texture
function generateStone() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(4);

    ctx.fillStyle = '#7a7a7a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Add stone cracks and variation
    for (let i = 0; i < 60; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 40) - 20;
        ctx.fillStyle = `rgb(${122+shade}, ${122+shade}, ${122+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    // Dark cracks
    for (let i = 0; i < 15; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        ctx.fillStyle = `rgb(${80+Math.floor(rand()*20)}, ${80+Math.floor(rand()*20)}, ${80+Math.floor(rand()*20)})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.06);
    return canvas;
}

// Wood log side texture
function generateWoodLogSide() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(5);

    ctx.fillStyle = '#6b5030';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Bark lines
    for (let x = 0; x < TEXTURE_SIZE; x += 3) {
        ctx.fillStyle = '#5a4028';
        ctx.fillRect(x, 0, 1, TEXTURE_SIZE);
    }

    // Vertical streaks
    for (let y = 0; y < TEXTURE_SIZE; y++) {
        const offset = Math.floor(rand() * 2);
        ctx.fillStyle = '#7a5a35';
        ctx.fillRect(offset, y, 1, 1);
    }

    addNoise(ctx, rand, 0.08);
    return canvas;
}

// Wood log top texture
function generateWoodLogTop() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(6);

    // Outer bark
    ctx.fillStyle = '#6b5030';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Inner wood rings
    ctx.fillStyle = '#b8945f';
    ctx.fillRect(1, 1, 14, 14);

    // Ring pattern
    const cx = 8, cy = 8;
    for (let r = 2; r < 7; r += 2) {
        ctx.strokeStyle = '#a07840';
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Leaves texture
function generateLeaves() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(7);

    ctx.fillStyle = '#3a8b2a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Leaf clusters
    for (let i = 0; i < 80; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 40) - 20;
        const alpha = rand() * 0.5 + 0.5;
        ctx.fillStyle = `rgba(${58+shade}, ${139+shade}, ${42+shade}, ${alpha})`;
        ctx.fillRect(x, y, 1, 1);
    }

    // Some transparent spots
    for (let i = 0; i < 10; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        ctx.clearRect(x, y, 1, 1);
    }

    return canvas;
}

// Sand texture
function generateSand() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(8);

    ctx.fillStyle = '#e8d9a0';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    for (let i = 0; i < 80; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 30) - 15;
        ctx.fillStyle = `rgb(${232+shade}, ${217+shade}, ${160+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.04);
    return canvas;
}

// Water texture
function generateWater() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(9);

    ctx.fillStyle = 'rgba(30, 100, 180, 0.7)';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Water caustics
    for (let i = 0; i < 30; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 30);
        ctx.fillStyle = `rgba(${50+shade}, ${120+shade}, ${200+shade}, 0.6)`;
        ctx.fillRect(x, y, 1, 1);
    }

    return canvas;
}

// Bedrock texture
function generateBedrock() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(10);

    ctx.fillStyle = '#333333';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    for (let i = 0; i < 100; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 60);
        ctx.fillStyle = `rgb(${30+shade}, ${30+shade}, ${30+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.1);
    return canvas;
}

// Wood plank texture
function generateWoodPlank() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(11);

    ctx.fillStyle = '#b8945f';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Plank lines
    ctx.fillStyle = '#9a7840';
    ctx.fillRect(0, 7, TEXTURE_SIZE, 1);
    ctx.fillRect(0, 15, TEXTURE_SIZE, 1);

    // Wood grain
    for (let i = 0; i < 40; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 20) - 10;
        ctx.fillStyle = `rgb(${184+shade}, ${148+shade}, ${95+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Cobblestone texture
function generateCobblestone() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(12);

    ctx.fillStyle = '#6a6a6a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Random cobble shapes
    for (let i = 0; i < 8; i++) {
        const x = Math.floor(rand() * 12);
        const y = Math.floor(rand() * 12);
        const w = Math.floor(rand() * 4) + 3;
        const h = Math.floor(rand() * 4) + 3;
        const shade = Math.floor(rand() * 40) - 20;
        ctx.fillStyle = `rgb(${106+shade}, ${106+shade}, ${106+shade})`;
        ctx.fillRect(x, y, w, h);

        // Dark outline
        ctx.strokeStyle = '#404040';
        ctx.strokeRect(x, y, w, h);
    }

    addNoise(ctx, rand, 0.08);
    return canvas;
}

// Ore texture generator
function generateOre(baseColor, oreColor, seed) {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(seed);

    // Stone base
    ctx.fillStyle = baseColor;
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Stone texture
    for (let i = 0; i < 40; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 30) - 15;
        ctx.fillStyle = `rgb(${122+shade}, ${122+shade}, ${122+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    // Ore spots
    const orePositions = [
        [3, 3], [8, 2], [12, 5], [5, 8], [10, 10], [2, 12], [13, 13]
    ];

    for (const [ox, oy] of orePositions) {
        if (rand() > 0.4) {
            const x = ox + Math.floor(rand() * 2);
            const y = oy + Math.floor(rand() * 2);
            ctx.fillStyle = oreColor;
            ctx.fillRect(x, y, 2, 2);
        }
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Coal ore
function generateCoalOre() {
    return generateOre('#7a7a7a', '#2a2a2a', 13);
}

// Iron ore
function generateIronOre() {
    return generateOre('#7a7a7a', '#c9a080', 14);
}

// Gold ore
function generateGoldOre() {
    return generateOre('#7a7a7a', '#fcee4b', 15);
}

// Diamond ore
function generateDiamondOre() {
    return generateOre('#7a7a7a', '#5decf5', 16);
}

// Torch texture
function generateTorch() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Stick
    ctx.fillStyle = '#8b6b4a';
    ctx.fillRect(7, 6, 2, 10);

    // Coal tip
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(7, 4, 2, 3);

    // Flame glow
    ctx.fillStyle = '#ffcc00';
    ctx.fillRect(6, 2, 4, 3);
    ctx.fillStyle = '#ff8800';
    ctx.fillRect(7, 1, 2, 2);

    return canvas;
}

// Glass texture
function generateGlass() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(17);

    ctx.fillStyle = 'rgba(200, 230, 255, 0.3)';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Frame
    ctx.strokeStyle = 'rgba(180, 210, 235, 0.8)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, 15, 15);

    // Light reflection
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(2, 2);
    ctx.lineTo(6, 2);
    ctx.lineTo(2, 6);
    ctx.closePath();
    ctx.fill();

    return canvas;
}

// Crafting table top
function generateCraftingTableTop() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(18);

    // Wood base
    ctx.fillStyle = '#b8945f';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Grid pattern
    ctx.strokeStyle = '#8a6830';
    ctx.lineWidth = 1;

    // 3x3 crafting grid
    for (let i = 1; i < 3; i++) {
        ctx.beginPath();
        ctx.moveTo(i * 5, 0);
        ctx.lineTo(i * 5, 16);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(0, i * 5);
        ctx.lineTo(16, i * 5);
        ctx.stroke();
    }

    // Tool marks
    ctx.fillStyle = '#9a7840';
    for (let i = 0; i < 5; i++) {
        ctx.fillRect(Math.floor(rand() * 16), Math.floor(rand() * 16), 1, 1);
    }

    addNoise(ctx, rand, 0.04);
    return canvas;
}

// Crafting table side
function generateCraftingTableSide() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(19);

    ctx.fillStyle = '#b8945f';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Tools
    ctx.fillStyle = '#5a5a5a';
    ctx.fillRect(3, 3, 2, 10); // Chisel
    ctx.fillStyle = '#8b6b4a';
    ctx.fillRect(10, 5, 2, 8); // Handle

    // Plank lines
    ctx.fillStyle = '#9a7840';
    ctx.fillRect(0, 7, TEXTURE_SIZE, 1);
    ctx.fillRect(0, 15, TEXTURE_SIZE, 1);

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Snow texture
function generateSnow() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(20);

    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    for (let i = 0; i < 40; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 15);
        ctx.fillStyle = `rgb(${240-shade}, ${240-shade}, ${255-shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    addNoise(ctx, rand, 0.02);
    return canvas;
}

// Snow grass side
function generateSnowGrassSide() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(21);

    // Dirt base
    ctx.fillStyle = '#8b6b4a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Add dirt variation
    for (let i = 0; i < 20; i++) {
        const x = Math.floor(rand() * TEXTURE_SIZE);
        const y = Math.floor(rand() * TEXTURE_SIZE);
        const shade = Math.floor(rand() * 30) - 15;
        ctx.fillStyle = `rgb(${139+shade}, ${107+shade}, ${74+shade})`;
        ctx.fillRect(x, y, 1, 1);
    }

    // Snow top — 2px strip at the very top
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, TEXTURE_SIZE, 2);

    // Snow overhang — sparse
    for (let x = 0; x < TEXTURE_SIZE; x++) {
        if (rand() > 0.6) {
            ctx.fillStyle = '#e8e8e8';
            ctx.fillRect(x, 2, 1, 1);
        }
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Cactus texture
function generateCactusTexture() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    const rand = textureRandom(22);

    ctx.fillStyle = '#2a8b2a';
    ctx.fillRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    // Vertical lines
    for (let x = 0; x < TEXTURE_SIZE; x += 4) {
        ctx.fillStyle = '#228822';
        ctx.fillRect(x, 0, 1, TEXTURE_SIZE);
    }

    // Spines
    for (let y = 2; y < TEXTURE_SIZE; y += 4) {
        ctx.fillStyle = '#3a9b3a';
        ctx.fillRect(0, y, 2, 1);
        ctx.fillRect(14, y, 2, 1);
    }

    addNoise(ctx, rand, 0.05);
    return canvas;
}

// Stick texture — diagonal Minecraft style
function generateStick() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    const S = TEXTURE_SIZE;
    // Diagonal stick from bottom-left to upper-center
    const pixels = [
        [6,4,'#a07848'],[7,4,'#8b6538'],
        [6,5,'#b8945f'],[7,5,'#a07848'],
        [7,6,'#a07848'],[8,6,'#8b6538'],
        [7,7,'#b8945f'],[8,7,'#a07848'],
        [8,8,'#a07848'],[9,8,'#8b6538'],
        [8,9,'#b8945f'],[9,9,'#a07848'],
        [9,10,'#a07848'],[10,10,'#8b6538'],
        [9,11,'#b8945f'],[10,11,'#a07848'],
        [10,12,'#a07848'],[11,12,'#7a5530'],
    ];
    for (const [x,y,c] of pixels) { ctx.fillStyle = c; ctx.fillRect(x,y,1,1); }
    return canvas;
}

// Helper: draw a pixel-art handle (diagonal, bottom-right to center)
function drawHandle(ctx, ox, oy) {
    const handle = [
        [0,0,'#a07848'],[1,0,'#8b6538'],
        [0,1,'#b8945f'],[1,1,'#a07848'],
        [1,2,'#a07848'],[2,2,'#8b6538'],
        [1,3,'#b8945f'],[2,3,'#a07848'],
        [2,4,'#a07848'],[3,4,'#8b6538'],
        [2,5,'#b8945f'],[3,5,'#a07848'],
    ];
    for (const [x,y,c] of handle) { ctx.fillStyle = c; ctx.fillRect(ox+x,oy+y,1,1); }
}

// Helper: draw wooden plank grain on a region
function drawWoodGrain(ctx, x, y, w, h, baseColor, darkColor, lightColor) {
    ctx.fillStyle = baseColor;
    ctx.fillRect(x, y, w, h);
    // Edge highlight top-left
    ctx.fillStyle = lightColor;
    ctx.fillRect(x, y, w, 1);
    ctx.fillRect(x, y, 1, h);
    // Edge shadow bottom-right
    ctx.fillStyle = darkColor;
    ctx.fillRect(x, y+h-1, w, 1);
    ctx.fillRect(x+w-1, y, 1, h);
    // Inner grain lines
    const rand = textureRandom(x * 13 + y * 7 + w);
    for (let i = 0; i < Math.floor(w * h / 4); i++) {
        const px = x + Math.floor(rand() * w);
        const py = y + Math.floor(rand() * h);
        ctx.fillStyle = rand() > 0.5 ? darkColor : lightColor;
        ctx.fillRect(px, py, 1, 1);
    }
}

// Wooden Pickaxe — Minecraft diagonal style
function generateWoodenPickaxe() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    // Handle
    drawHandle(ctx, 8, 8);
    // Head — horizontal bar with two downward prongs
    drawWoodGrain(ctx, 2, 1, 10, 3, '#b8945f', '#8a6830', '#d4b07a');
    // Left prong
    drawWoodGrain(ctx, 2, 4, 2, 3, '#a88050', '#8a6830', '#c8a468');
    // Right prong
    drawWoodGrain(ctx, 10, 4, 2, 3, '#a88050', '#8a6830', '#c8a468');
    // Binding where handle meets head
    ctx.fillStyle = '#7a5530';
    ctx.fillRect(7, 3, 2, 2);
    return canvas;
}

// Wooden Axe
function generateWoodenAxe() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    // Handle
    drawHandle(ctx, 8, 8);
    // Axe head — L shape
    drawWoodGrain(ctx, 3, 1, 6, 3, '#b8945f', '#8a6830', '#d4b07a');
    drawWoodGrain(ctx, 3, 4, 4, 4, '#a88050', '#8a6830', '#c8a468');
    // Binding
    ctx.fillStyle = '#7a5530';
    ctx.fillRect(7, 4, 2, 2);
    return canvas;
}

// Wooden Sword — blade shape
function generateWoodenSword() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    // Handle
    drawHandle(ctx, 9, 10);
    // Guard
    ctx.fillStyle = '#7a5530';
    ctx.fillRect(6, 9, 6, 1);
    ctx.fillStyle = '#6b4520';
    ctx.fillRect(6, 10, 6, 1);
    // Blade — tapers to point
    drawWoodGrain(ctx, 7, 2, 3, 7, '#c8a468', '#a07840', '#dcc090');
    // Tip
    ctx.fillStyle = '#c8a468';
    ctx.fillRect(8, 1, 1, 1);
    // Edge highlight
    ctx.fillStyle = '#dcc090';
    ctx.fillRect(7, 2, 1, 6);
    // Dark edge
    ctx.fillStyle = '#a07840';
    ctx.fillRect(9, 2, 1, 6);
    return canvas;
}

// Wooden Shovel — rounded blade
function generateWoodenShovel() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    // Handle
    drawHandle(ctx, 8, 8);
    // Blade — rounded rectangle
    drawWoodGrain(ctx, 5, 2, 6, 6, '#b8945f', '#8a6830', '#d4b07a');
    // Round the top corners
    ctx.clearRect(5, 2, 1, 1);
    ctx.clearRect(10, 2, 1, 1);
    // Round bottom corners
    ctx.clearRect(5, 7, 1, 1);
    ctx.clearRect(10, 7, 1, 1);
    // Center line
    ctx.fillStyle = '#a07840';
    ctx.fillRect(8, 2, 1, 6);
    // Binding
    ctx.fillStyle = '#7a5530';
    ctx.fillRect(7, 7, 2, 2);
    return canvas;
}

// Wooden Hoe — two prongs
function generateWoodenHoe() {
    const canvas = createTextureCanvas();
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
    // Handle
    drawHandle(ctx, 8, 8);
    // Head — horizontal bar
    drawWoodGrain(ctx, 3, 2, 6, 2, '#b8945f', '#8a6830', '#d4b07a');
    // Left tooth
    drawWoodGrain(ctx, 3, 4, 2, 3, '#a88050', '#8a6830', '#c8a468');
    // Binding
    ctx.fillStyle = '#7a5530';
    ctx.fillRect(7, 3, 2, 2);
    return canvas;
}

// Create texture atlas
function createTextureAtlas() {
    // Generate textures with error handling
    const textures = {};

    try { textures[BlockType.GRASS] = {
        top: generateGrassTop(),
        side: generateGrassSide(),
        bottom: generateDirt()
    }; } catch(e) { console.error('Failed to generate GRASS texture', e); }

    try { textures[BlockType.DIRT] = generateDirt(); } catch(e) { console.error('Failed to generate DIRT texture', e); }
    try { textures[BlockType.STONE] = generateStone(); } catch(e) { console.error('Failed to generate STONE texture', e); }

    try { textures[BlockType.WOOD_LOG] = {
        top: generateWoodLogTop(),
        side: generateWoodLogSide(),
        bottom: generateWoodLogTop()
    }; } catch(e) { console.error('Failed to generate WOOD_LOG texture', e); }

    try { textures[BlockType.LEAVES] = generateLeaves(); } catch(e) { console.error('Failed to generate LEAVES texture', e); }
    try { textures[BlockType.SAND] = generateSand(); } catch(e) { console.error('Failed to generate SAND texture', e); }
    try { textures[BlockType.WATER] = generateWater(); } catch(e) { console.error('Failed to generate WATER texture', e); }
    try { textures[BlockType.BEDROCK] = generateBedrock(); } catch(e) { console.error('Failed to generate BEDROCK texture', e); }
    try { textures[BlockType.WOOD_PLANK] = generateWoodPlank(); } catch(e) { console.error('Failed to generate WOOD_PLANK texture', e); }
    try { textures[BlockType.COBBLESTONE] = generateCobblestone(); } catch(e) { console.error('Failed to generate COBBLESTONE texture', e); }
    try { textures[BlockType.COAL_ORE] = generateCoalOre(); } catch(e) { console.error('Failed to generate COAL_ORE texture', e); }
    try { textures[BlockType.IRON_ORE] = generateIronOre(); } catch(e) { console.error('Failed to generate IRON_ORE texture', e); }
    try { textures[BlockType.GOLD_ORE] = generateGoldOre(); } catch(e) { console.error('Failed to generate GOLD_ORE texture', e); }
    try { textures[BlockType.DIAMOND_ORE] = generateDiamondOre(); } catch(e) { console.error('Failed to generate DIAMOND_ORE texture', e); }
    try { textures[BlockType.TORCH] = generateTorch(); } catch(e) { console.error('Failed to generate TORCH texture', e); }
    try { textures[BlockType.GLASS] = generateGlass(); } catch(e) { console.error('Failed to generate GLASS texture', e); }

    try { textures[BlockType.CRAFTING_TABLE] = {
        top: generateCraftingTableTop(),
        side: generateCraftingTableSide(),
        bottom: generateWoodPlank()
    }; } catch(e) { console.error('Failed to generate CRAFTING_TABLE texture', e); }

    try { textures[BlockType.SNOW] = {
        top: generateSnow(),
        side: generateSnowGrassSide(),
        bottom: generateDirt()
    }; } catch(e) { console.error('Failed to generate SNOW texture', e); }
    try { textures[BlockType.CACTUS] = generateCactusTexture(); } catch(e) { console.error('Failed to generate CACTUS texture', e); }

    // Tool/item textures
    try { textures[BlockType.STICK] = generateStick(); } catch(e) { console.error('Failed to generate STICK texture', e); }
    try { textures[BlockType.WOODEN_PICKAXE] = generateWoodenPickaxe(); } catch(e) { console.error('Failed to generate WOODEN_PICKAXE texture', e); }
    try { textures[BlockType.WOODEN_AXE] = generateWoodenAxe(); } catch(e) { console.error('Failed to generate WOODEN_AXE texture', e); }
    try { textures[BlockType.WOODEN_SWORD] = generateWoodenSword(); } catch(e) { console.error('Failed to generate WOODEN_SWORD texture', e); }
    try { textures[BlockType.WOODEN_SHOVEL] = generateWoodenShovel(); } catch(e) { console.error('Failed to generate WOODEN_SHOVEL texture', e); }
    try { textures[BlockType.WOODEN_HOE] = generateWoodenHoe(); } catch(e) { console.error('Failed to generate WOODEN_HOE texture', e); }

    // Calculate atlas size (arrange in grid)
    const atlasSize = 256; // 16 textures per row, 16x16 each
    textureCanvas = document.createElement('canvas');
    textureCanvas.width = atlasSize;
    textureCanvas.height = atlasSize;
    const ctx = textureCanvas.getContext('2d');

    let index = 0;
    const texturesPerRow = atlasSize / TEXTURE_SIZE;

    function addTextureToAtlas(tex, blockType, face = 'all') {
        const x = (index % texturesPerRow) * TEXTURE_SIZE;
        const y = Math.floor(index / texturesPerRow) * TEXTURE_SIZE;

        ctx.drawImage(tex, x, y);

        const u0 = x / atlasSize;
        const v0 = y / atlasSize;
        const u1 = (x + TEXTURE_SIZE) / atlasSize;
        const v1 = (y + TEXTURE_SIZE) / atlasSize;

        const key = String(blockType); // Ensure consistent string key
        if (!textureUVs[key]) {
            textureUVs[key] = {};
        }

        textureUVs[key][face] = { u0, v0, u1, v1 };
        index++;
    }

    // Add all textures to atlas
    for (const [blockTypeStr, texture] of Object.entries(textures)) {
        const blockType = parseInt(blockTypeStr); // Convert back to number

        if (!texture) {
            console.warn(`Missing texture for block type ${blockType}`);
            continue;
        }

        if (texture instanceof HTMLCanvasElement) {
            // Single texture for all faces
            addTextureToAtlas(texture, blockType, 'all');
            const key = String(blockType);
            textureUVs[key].top = textureUVs[key].all;
            textureUVs[key].bottom = textureUVs[key].all;
            textureUVs[key].side = textureUVs[key].all;
        } else if (typeof texture === 'object') {
            // Different textures per face
            if (!texture.top) {
                console.warn(`Missing top texture for block type ${blockType}`);
                continue;
            }
            if (!texture.side) {
                console.warn(`Missing side texture for block type ${blockType}`);
                continue;
            }
            if (!texture.bottom) {
                console.warn(`Missing bottom texture for block type ${blockType}`);
                continue;
            }
            addTextureToAtlas(texture.top, blockType, 'top');
            addTextureToAtlas(texture.bottom, blockType, 'bottom');
            addTextureToAtlas(texture.side, blockType, 'side');
        } else {
            console.warn(`Invalid texture structure for block type ${blockType}`, texture);
        }
    }

    // Create Three.js texture
    // IMPORTANT: flipY must be false because our UV coordinates use
    // canvas coordinate system (y=0 at top), not OpenGL (y=0 at bottom)
    textureAtlas = new THREE.CanvasTexture(textureCanvas);
    textureAtlas.flipY = false;
    textureAtlas.magFilter = THREE.NearestFilter;
    textureAtlas.minFilter = THREE.NearestFilter;
    textureAtlas.wrapS = THREE.RepeatWrapping;
    textureAtlas.wrapT = THREE.RepeatWrapping;

    return textureAtlas;
}

// Get UV coordinates for block face
function getBlockUVs(blockType, face) {
    const key = String(blockType); // Use string key for consistency
    const blockUVs = textureUVs[key];
    if (!blockUVs) return { u0: 0, v0: 0, u1: 1, v1: 1 };

    if (blockUVs[face]) return blockUVs[face];
    if (blockUVs.all) return blockUVs.all;
    if (blockUVs.side) return blockUVs.side;

    return { u0: 0, v0: 0, u1: 1, v1: 1 };
}

// Get texture canvas for inventory display
function getBlockIcon(blockType) {
    // Ensure texture atlas exists
    if (!textureCanvas) {
        createTextureAtlas();
    }

    const uvs = getBlockUVs(blockType, 'side');

    const canvas = document.createElement('canvas');
    canvas.width = TEXTURE_SIZE;
    canvas.height = TEXTURE_SIZE;
    const ctx = canvas.getContext('2d');

    // Copy from atlas
    const sx = uvs.u0 * textureCanvas.width;
    const sy = uvs.v0 * textureCanvas.height;

    ctx.drawImage(textureCanvas, sx, sy, TEXTURE_SIZE, TEXTURE_SIZE, 0, 0, TEXTURE_SIZE, TEXTURE_SIZE);

    return canvas;
}

// Block properties
const STONE_HARDNESS = 1.5;

const BlockProperties = {
    [BlockType.AIR]: { solid: false, transparent: true },
    [BlockType.GRASS]: { solid: true, transparent: false, hardness: 0.6, harvestTool: 'shovel' },
    [BlockType.DIRT]: { solid: true, transparent: false, hardness: 0.5, harvestTool: 'shovel' },
    [BlockType.STONE]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'pickaxe' },
    [BlockType.WOOD_LOG]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'axe' },
    [BlockType.LEAVES]: { solid: true, transparent: true, hardness: 0.2 },
    [BlockType.SAND]: { solid: true, transparent: false, hardness: 0.5, harvestTool: 'shovel' },
    [BlockType.WATER]: { solid: false, transparent: true, liquid: true },
    [BlockType.BEDROCK]: { solid: true, transparent: false, hardness: -1 },
    [BlockType.WOOD_PLANK]: { solid: true, transparent: false, hardness: 2.0, harvestTool: 'axe' },
    [BlockType.COBBLESTONE]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'pickaxe' },
    [BlockType.COAL_ORE]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'pickaxe' },
    [BlockType.IRON_ORE]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'pickaxe' },
    [BlockType.GOLD_ORE]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'pickaxe' },
    [BlockType.DIAMOND_ORE]: { solid: true, transparent: false, hardness: STONE_HARDNESS, harvestTool: 'pickaxe' },
    [BlockType.TORCH]: { solid: false, transparent: true, hardness: 0.0, light: 14 },
    [BlockType.GLASS]: { solid: true, transparent: true, hardness: 0.3 },
    [BlockType.CRAFTING_TABLE]: { solid: true, transparent: false, hardness: 2.5 },
    [BlockType.SNOW]: { solid: true, transparent: false, hardness: 0.2 },
    [BlockType.CACTUS]: { solid: true, transparent: false, hardness: 0.4 },
    // Tools/items — not placeable
    [BlockType.STICK]: { solid: false, transparent: true, isItem: true, maxStack: 64 },
    [BlockType.WOODEN_PICKAXE]: { solid: false, transparent: true, isItem: true, maxStack: 1, toolSpeed: 3.0, toolType: 'pickaxe' },
    [BlockType.WOODEN_AXE]: { solid: false, transparent: true, isItem: true, maxStack: 1, toolSpeed: 3.0, toolType: 'axe' },
    [BlockType.WOODEN_SWORD]: { solid: false, transparent: true, isItem: true, maxStack: 1, toolSpeed: 1.5, toolType: 'sword', damage: 4 },
    [BlockType.WOODEN_SHOVEL]: { solid: false, transparent: true, isItem: true, maxStack: 1, toolSpeed: 3.0, toolType: 'shovel' },
    [BlockType.WOODEN_HOE]: { solid: false, transparent: true, isItem: true, maxStack: 1, toolSpeed: 1.0, toolType: 'hoe' }
};

function isBlockSolid(blockType) {
    const props = BlockProperties[blockType];
    return props ? props.solid : false;
}

function isBlockTransparent(blockType) {
    const props = BlockProperties[blockType];
    return props ? props.transparent : true;
}
