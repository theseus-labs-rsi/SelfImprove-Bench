/**
 * Simplex Noise Implementation
 * Used for procedural terrain generation
 */

class SimplexNoise {
    constructor(seed = Math.random()) {
        this.p = new Uint8Array(256);
        this.perm = new Uint8Array(512);
        this.permMod12 = new Uint8Array(512);

        // Seeded random
        this.seed = seed;
        const random = this.seededRandom(seed);

        // Initialize permutation table
        for (let i = 0; i < 256; i++) {
            this.p[i] = i;
        }

        // Shuffle
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [this.p[i], this.p[j]] = [this.p[j], this.p[i]];
        }

        // Extend permutation table
        for (let i = 0; i < 512; i++) {
            this.perm[i] = this.p[i & 255];
            this.permMod12[i] = this.perm[i] % 12;
        }

        // Gradient vectors for 3D
        this.grad3 = new Float32Array([
            1,1,0, -1,1,0, 1,-1,0, -1,-1,0,
            1,0,1, -1,0,1, 1,0,-1, -1,0,-1,
            0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1
        ]);

        // Skewing factors for 3D
        this.F3 = 1.0 / 3.0;
        this.G3 = 1.0 / 6.0;

        // Skewing factors for 2D
        this.F2 = 0.5 * (Math.sqrt(3.0) - 1.0);
        this.G2 = (3.0 - Math.sqrt(3.0)) / 6.0;
    }

    seededRandom(seed) {
        return function() {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };
    }

    // 2D Simplex Noise
    noise2D(x, y) {
        const F2 = this.F2;
        const G2 = this.G2;

        // Skew input space
        const s = (x + y) * F2;
        const i = Math.floor(x + s);
        const j = Math.floor(y + s);

        const t = (i + j) * G2;
        const X0 = i - t;
        const Y0 = j - t;
        const x0 = x - X0;
        const y0 = y - Y0;

        // Determine which simplex
        let i1, j1;
        if (x0 > y0) {
            i1 = 1; j1 = 0;
        } else {
            i1 = 0; j1 = 1;
        }

        const x1 = x0 - i1 + G2;
        const y1 = y0 - j1 + G2;
        const x2 = x0 - 1.0 + 2.0 * G2;
        const y2 = y0 - 1.0 + 2.0 * G2;

        const ii = i & 255;
        const jj = j & 255;

        // Calculate contribution from corners
        let n0 = 0, n1 = 0, n2 = 0;

        let t0 = 0.5 - x0*x0 - y0*y0;
        if (t0 >= 0) {
            const gi0 = this.permMod12[ii + this.perm[jj]];
            t0 *= t0;
            n0 = t0 * t0 * this.dot2(gi0, x0, y0);
        }

        let t1 = 0.5 - x1*x1 - y1*y1;
        if (t1 >= 0) {
            const gi1 = this.permMod12[ii + i1 + this.perm[jj + j1]];
            t1 *= t1;
            n1 = t1 * t1 * this.dot2(gi1, x1, y1);
        }

        let t2 = 0.5 - x2*x2 - y2*y2;
        if (t2 >= 0) {
            const gi2 = this.permMod12[ii + 1 + this.perm[jj + 1]];
            t2 *= t2;
            n2 = t2 * t2 * this.dot2(gi2, x2, y2);
        }

        // Scale to [-1, 1]
        return 70.0 * (n0 + n1 + n2);
    }

    // 3D Simplex Noise
    noise3D(x, y, z) {
        const F3 = this.F3;
        const G3 = this.G3;

        // Skew input space
        const s = (x + y + z) * F3;
        const i = Math.floor(x + s);
        const j = Math.floor(y + s);
        const k = Math.floor(z + s);

        const t = (i + j + k) * G3;
        const X0 = i - t;
        const Y0 = j - t;
        const Z0 = k - t;
        const x0 = x - X0;
        const y0 = y - Y0;
        const z0 = z - Z0;

        // Determine which simplex
        let i1, j1, k1;
        let i2, j2, k2;

        if (x0 >= y0) {
            if (y0 >= z0) {
                i1=1; j1=0; k1=0; i2=1; j2=1; k2=0;
            } else if (x0 >= z0) {
                i1=1; j1=0; k1=0; i2=1; j2=0; k2=1;
            } else {
                i1=0; j1=0; k1=1; i2=1; j2=0; k2=1;
            }
        } else {
            if (y0 < z0) {
                i1=0; j1=0; k1=1; i2=0; j2=1; k2=1;
            } else if (x0 < z0) {
                i1=0; j1=1; k1=0; i2=0; j2=1; k2=1;
            } else {
                i1=0; j1=1; k1=0; i2=1; j2=1; k2=0;
            }
        }

        const x1 = x0 - i1 + G3;
        const y1 = y0 - j1 + G3;
        const z1 = z0 - k1 + G3;
        const x2 = x0 - i2 + 2.0*G3;
        const y2 = y0 - j2 + 2.0*G3;
        const z2 = z0 - k2 + 2.0*G3;
        const x3 = x0 - 1.0 + 3.0*G3;
        const y3 = y0 - 1.0 + 3.0*G3;
        const z3 = z0 - 1.0 + 3.0*G3;

        const ii = i & 255;
        const jj = j & 255;
        const kk = k & 255;

        // Calculate contributions
        let n0 = 0, n1 = 0, n2 = 0, n3 = 0;

        let t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
        if (t0 >= 0) {
            const gi0 = this.permMod12[ii + this.perm[jj + this.perm[kk]]];
            t0 *= t0;
            n0 = t0 * t0 * this.dot3(gi0, x0, y0, z0);
        }

        let t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
        if (t1 >= 0) {
            const gi1 = this.permMod12[ii + i1 + this.perm[jj + j1 + this.perm[kk + k1]]];
            t1 *= t1;
            n1 = t1 * t1 * this.dot3(gi1, x1, y1, z1);
        }

        let t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
        if (t2 >= 0) {
            const gi2 = this.permMod12[ii + i2 + this.perm[jj + j2 + this.perm[kk + k2]]];
            t2 *= t2;
            n2 = t2 * t2 * this.dot3(gi2, x2, y2, z2);
        }

        let t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
        if (t3 >= 0) {
            const gi3 = this.permMod12[ii + 1 + this.perm[jj + 1 + this.perm[kk + 1]]];
            t3 *= t3;
            n3 = t3 * t3 * this.dot3(gi3, x3, y3, z3);
        }

        // Scale to [-1, 1]
        return 32.0 * (n0 + n1 + n2 + n3);
    }

    dot2(gi, x, y) {
        const g = this.grad3;
        return g[gi*3] * x + g[gi*3+1] * y;
    }

    dot3(gi, x, y, z) {
        const g = this.grad3;
        return g[gi*3] * x + g[gi*3+1] * y + g[gi*3+2] * z;
    }

    // Fractal Brownian Motion (layered noise)
    fbm2D(x, y, octaves = 4, lacunarity = 2.0, persistence = 0.5) {
        let value = 0;
        let amplitude = 1.0;
        let frequency = 1.0;
        let maxValue = 0;

        for (let i = 0; i < octaves; i++) {
            value += amplitude * this.noise2D(x * frequency, y * frequency);
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return value / maxValue;
    }

    fbm3D(x, y, z, octaves = 4, lacunarity = 2.0, persistence = 0.5) {
        let value = 0;
        let amplitude = 1.0;
        let frequency = 1.0;
        let maxValue = 0;

        for (let i = 0; i < octaves; i++) {
            value += amplitude * this.noise3D(x * frequency, y * frequency, z * frequency);
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return value / maxValue;
    }

    // Ridged noise for mountains
    ridged2D(x, y, octaves = 4, lacunarity = 2.0, persistence = 0.5) {
        let value = 0;
        let amplitude = 1.0;
        let frequency = 1.0;
        let weight = 1.0;

        for (let i = 0; i < octaves; i++) {
            let signal = this.noise2D(x * frequency, y * frequency);
            signal = 1.0 - Math.abs(signal);
            signal *= signal * weight;
            weight = Math.min(1, Math.max(0, signal * 2));
            value += signal * amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        return value;
    }
}

// Global noise instance
let noise = null;

function initNoise(seed) {
    noise = new SimplexNoise(seed);
}
