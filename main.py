# original code: https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html

import numpy as np
import taichi as ti


ti.init()

class Scene:
    gravity = -9.81
    dt = 1.0 / 120.0
    flipRatio = 0.9
    numPressureIters = 100
    numParticleIters = 2
    frameNr = 0
    overRelaxation = 1.9
    compensateDrift = True
    separateParticles = True
    obstacleX = 0.0
    obstacleY = 0.0
    obstacleRadius = 0.15
    paused = True
    showObstacle = True
    obstacleVelX = 0.0
    obstacleVelY = 0.0
    showParticles = True
    showGrid = False
    fluid = None


scene = Scene()


simHeight = 3.0
cScale = 1.0
simWidth = 3.0

U_FIELD = 0
V_FIELD = 1

FLUID_CELL = 0
AIR_CELL = 1
SOLID_CELL = 2

cnt = 0

def clamp(x, min, max):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x



@ti.kernel
def solveIncompressibility_kernel(n:int, cp:float, fNumX:ti.i32, fNumY:ti.i32, fInvSpacing:ti.f32, u:ti.types.ndarray(), v:ti.types.ndarray(), p:ti.types.ndarray(), s:ti.types.ndarray(), cellType:ti.types.ndarray(), particleRestDensity:ti.f32, particleDensity:ti.types.ndarray(), overRelaxation:ti.f32, compensateDrift:ti.i32):
    for i in range(1, fNumX - 1):
        for j in range(1, fNumY - 1):
            if cellType[i * n + j] != FLUID_CELL:
                continue

            center = i * n + j
            left = (i - 1) * n + j
            right = (i + 1) * n + j
            bottom = i * n + j - 1
            top = i * n + j + 1

            sc = s[center]
            sx0 = s[left]
            sx1 = s[right]
            sy0 = s[bottom]
            sy1 = s[top]
            sc = (sx0 + sx1 + sy0 + sy1)

            if sc == 0.0:
                continue

            div = u[right] - u[center] + v[top] - v[center]

            if particleRestDensity > 0.0 and compensateDrift:
                k = 1.0
                compression = particleDensity[i * n + j] - particleRestDensity
                if compression > 0.0:
                    div = div - k * compression
            
            pc = -div / sc
            pc *= overRelaxation
            p[center] += cp * pc

            u[center] -= sx0 * pc
            u[right] += sx1 * pc
            v[center] -= sy0 * pc
            v[top] += sy1 * pc

@ti.kernel
def pushParticlesApart_kernel(particleRadius:float, colorDiffusionCoeff:float, numIters:int,
                               numParticles:int, pNumX:ti.i32, pNumY:ti.i32, pInvSpacing:ti.f32,
                               particlePos:ti.types.ndarray(), firstCellParticle:ti.types.ndarray(),
                                 cellParticleIds:ti.types.ndarray(), particleColor:ti.types.ndarray(), ):
    minDist = 2.0 * particleRadius
    minDist2 = minDist * minDist
    for iter in range(numIters):
        for i in range(numParticles):
            px = particlePos[2 * i]
            py = particlePos[2 * i + 1]

            pxi = ti.math.clamp(ti.floor(px * pInvSpacing), 0, pNumX - 1)
            pyi = ti.math.clamp(ti.floor(py * pInvSpacing), 0, pNumY - 1)
            x0 = ti.math.max(pxi - 1, 0)
            y0 = ti.math.max(pyi - 1, 0)
            x1 = ti.math.min(pxi + 1, pNumX - 1)
            y1 = ti.math.min(pyi + 1, pNumY - 1)
            x0 = int(x0)
            y0 = int(y0)
            x1 = int(x1)
            y1 = int(y1)

            for xi in range(x0, x1 + 1):
                for yi in range(y0, y1+1):
                    cellNr = xi * pNumY + yi
                    first = firstCellParticle[cellNr]
                    last = firstCellParticle[cellNr + 1]
                    for j in range(first, last):
                        id = cellParticleIds[j]
                        if id == i:
                            continue
                        qx = particlePos[2 * id]
                        qy = particlePos[2 * id + 1]

                        dx = qx - px
                        dy = qy - py
                        d2 = dx * dx + dy * dy

                        if (d2 > minDist2 or d2 == 0.0):
                            continue

                        d = ti.sqrt(d2)
                        s = 0.5 * (minDist - d) / d
                        dx *= s
                        dy *= s
                        particlePos[2 * i] -= dx
                        particlePos[2 * i + 1] -= dy
                        particlePos[2 * id] += dx
                        particlePos[2 * id + 1] += dy

                        # diffuse colors
                        for k in range(3):
                            color0 = particleColor[3 * i + k]
                            color1 = particleColor[3 * id + k]
                            color = 0.5 * (color0 + color1)
                            particleColor[3 * i + k] = color0 + colorDiffusionCoeff * (color - color0)
                            particleColor[3 * id + k] = color1 + colorDiffusionCoeff * (color - color1)

@ti.kernel
def pushParticlesApart_countParticlesPerCell_kernel(numParticles:int, pNumX:ti.i32, pNumY:ti.i32,
                                                     pInvSpacing:ti.f32, particlePos:ti.types.ndarray(), numCellParticles:ti.types.ndarray()):
    for i in range(numParticles):
        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]

        xi = int(ti.math.clamp(ti.math.floor(x * pInvSpacing), 0, pNumX - 1))
        yi = int(ti.math.clamp(ti.math.floor(y * pInvSpacing), 0, pNumY - 1))
        cellNr = int(xi * pNumY + yi)
        numCellParticles[cellNr] += 1

@ti.kernel  
def pushParticlesApart_fillCellParticleIds_kernel(numParticles:int, pNumX:ti.i32, pNumY:ti.i32,pInvSpacing:ti.f32,
                                                    particlePos:ti.types.ndarray(), firstCellParticle:ti.types.ndarray(),
                                                    cellParticleIds:ti.types.ndarray()):
    for i in range(numParticles):
        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]

        xi = int(ti.math.clamp(ti.math.floor(x * pInvSpacing), 0, pNumX - 1))
        yi = int(ti.math.clamp(ti.math.floor(y * pInvSpacing), 0, pNumY - 1))

        cellNr = int(xi * pNumY + yi)
        firstCellParticle[cellNr] -= 1
        cellParticleIds[firstCellParticle[cellNr]] = i

@ti.kernel
def pushParticlesApart_partialSums_kernel(pNumCells:int, numCellParticles:ti.types.ndarray(),
                                            firstCellParticle:ti.types.ndarray()):
    first = 0
    for i in range(pNumCells):
        first += numCellParticles[i]
        firstCellParticle[i] = first
    firstCellParticle[pNumCells] = first # guard


@ti.kernel
def transferVelocities_kernel(n:ti.i32, numParticles:int,h:ti.f32, h1:ti.f32, dx:ti.f32, dy:ti.f32, fNumX:ti.i32, fNumY:ti.i32,
                                particlePos:ti.types.ndarray(), particleVel:ti.types.ndarray(), 
                                  f:ti.types.ndarray(), toGrid:ti.i32, component:ti.i32, cellType:ti.types.ndarray(),
                                  prevF:ti.types.ndarray(), d:ti.types.ndarray(), flipRatio:ti.f32, ):
    for i in range(numParticles):
        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]

        x = ti.math.clamp(x, h, (fNumX - 1) * h)
        y = ti.math.clamp(y, h, (fNumY - 1) * h)

        x0 = ti.math.min(ti.math.floor((x - dx) * h1), fNumX - 2)
        x1 = ti.math.min(x0 + 1, fNumX - 2)
        x0 = int(x0)
        x1 = int(x1)
        tx = (x - dx) - x0 * h

        y0 = ti.math.min(ti.math.floor((y - dy) * h1), fNumY - 2)
        y1 = ti.math.min(y0 + 1, fNumY - 2)
        y0 = int(y0)
        y1 = int(y1)
        ty = ((y - dy) - y0*h) * h1

        sx = 1.0 - tx
        sy = 1.0 - ty

        d0 = sx * sy
        d1 = tx * sy
        d2 = tx * ty
        d3 = sx * ty

        nr0 = int(x0 * n + y0)
        nr1 = int(x1 * n + y0)
        nr2 = int(x1 * n + y1)
        nr3 = int(x0 * n + y1)



        if toGrid:
            pv = particleVel[2 * i + component]
            f[nr0] += pv * d0
            d[nr0] += d0
            f[nr1] += pv * d1
            d[nr1] += d1
            f[nr2] += pv * d2
            d[nr2] += d2
            f[nr3] += pv * d3
            d[nr3] += d3

        else:
            offset = n if component == 0 else 1
            valid0 = 1.0 if cellType[nr0] != AIR_CELL or cellType[nr0 - offset] != AIR_CELL else 0.0
            valid1 = 1.0 if cellType[nr1] != AIR_CELL or cellType[nr1 - offset] != AIR_CELL else 0.0
            valid2 = 1.0 if cellType[nr2] != AIR_CELL or cellType[nr2 - offset] != AIR_CELL else 0.0
            valid3 = 1.0 if cellType[nr3] != AIR_CELL or cellType[nr3 - offset] != AIR_CELL else 0.0

            v_ = particleVel[2 * i + component]
            d_ = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3

            if d_ > 0.0:
                picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / d_
                corr = (valid0 * d0 * (f[nr0] - prevF[nr0]) + valid1 * d1 * (f[nr1] - prevF[nr1]) + valid2 * d2 * (f[nr2] - prevF[nr2]) + valid3 * d3 * (f[nr3] - prevF[nr3])) / d_
                flipV = v_ + corr
                particleVel[2 * i + component] = (1.0 - flipRatio) * picV + flipRatio * flipV

@ti.kernel
def transferVelocities_setCellType_kernel(n:ti.i32, numParticles:int, fNumX:ti.i32, fNumY:ti.i32, h1:ti.f32, cellType:ti.types.ndarray(), fNumCells:int,  particlePos:ti.types.ndarray(), s:ti.types.ndarray()):
    for i in range(fNumCells):
        cellType[i] = SOLID_CELL if s[i] == 0.0 else AIR_CELL

    for i in range(numParticles):
        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]
        xi = int(ti.math.clamp(ti.math.floor(x * h1), 0, fNumX - 1))
        yi = int(ti.math.clamp(ti.math.floor(y * h1), 0, fNumY - 1))
        cellNr = int(xi * n + yi)
        if cellType[cellNr] == AIR_CELL:
            cellType[cellNr] = FLUID_CELL

@ti.kernel
def transferVelocities_restoreSolidCells_kernel(fNumX:int, fNumY:int, n:int, cellType:ti.types.ndarray(), u:ti.types.ndarray(), v:ti.types.ndarray(), prevU:ti.types.ndarray(), prevV:ti.types.ndarray(),f:ti.types.ndarray(),d:ti.types.ndarray()):
    for i in range(f.shape[0]):
        if d[i] > 0.0:
            f[i] /= d[i]
    for i in range(fNumX):
        for j in range(fNumY):
            solid = cellType[i * n + j] == SOLID_CELL
            if solid or (i > 0 and cellType[(i - 1) * n + j] == SOLID_CELL):
                u[i * n + j] = prevU[i * n + j]
            if solid or (j > 0 and cellType[i * n + j - 1] == SOLID_CELL):
                v[i * n + j] = prevV[i * n + j]



@ti.kernel
def updateParticleDensity_kernel(numParticles:int,particlePos:ti.types.ndarray(),
                                 h:float,h1:float,h2:float,fNumX:int,fNumY:int,
                                d:ti.types.ndarray(), n:int):
    for i in range(numParticles):
        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]

        x = ti.math.clamp(x, h, (fNumX - 1) * h)
        y = ti.math.clamp(y, h, (fNumY - 1) * h)

        x0 = int(ti.math.floor((x - h2) * h1))
        x1 = int(ti.math.min(x0 + 1, fNumX - 2))
        tx = (x - h2) - x0 * h

        y0 = int(ti.math.floor((y - h2) * h1))
        y1 = int(ti.math.min(y0 + 1, fNumY - 2))
        ty = (y - h2) - y0 * h

        sx = 1.0 - tx
        sy = 1.0 - ty

        # x0 = int(x0)
        # x1 = int(x1)
        # y0 = int(y0)
        # y1 = int(y1)

        if x0 < fNumX and y0 < fNumY:
            d[x0 * n + y0] += sx * sy
        if x1 < fNumX and y0 < fNumY:
            d[x1 * n + y0] += tx * sy
        if x1 < fNumX and y1 < fNumY:
            d[x1 * n + y1] += tx * ty
        if x0 < fNumX and y1 < fNumY:
            d[x0 * n + y1] += sx * ty

@ti.kernel
def updateParticleColors_kernel(numParticles:int,particlePos:ti.types.ndarray(),particleRestDensity:float,particleDensity:ti.types.ndarray(),
                                fNumX:int,fNumY:int,particleColor:ti.types.ndarray(), fInvSpacing:float, h :float):
    h1 = fInvSpacing
    
    for i in range(numParticles):
        s = 0.01

        particleColor[3 * i] = ti.math.clamp(particleColor[3 * i] - s, 0.0, 1.0)
        particleColor[3 * i + 1] = ti.math.clamp(particleColor[3 * i + 1] - s, 0.0, 1.0)
        particleColor[3 * i + 2] = ti.math.clamp(particleColor[3 * i + 2] + s, 0.0, 1.0)

        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]
        xi = ti.math.clamp(ti.math.floor((x - 0.5 * h) * h1), 0, fNumX - 1)
        yi = ti.math.clamp(ti.math.floor((y - 0.5 * h) * h1), 0, fNumY - 1)
        cellNr = int(xi * fNumY + yi)

        d0 = particleRestDensity

        if d0 > 0.0:
            relDensity = particleDensity[cellNr] / d0
            if relDensity < 0.7:
                s = 0.8
                particleColor[3 * i] = s
                particleColor[3 * i + 1] = s
                particleColor[3 * i + 2] = 1.0

@ti.kernel
def updateCellColors_kernel(cellColor:ti.types.ndarray(),cellType:ti.types.ndarray(),particleDensity:ti.types.ndarray(),particleRestDensity:float,fNumCells:int):
    for i in range(fNumCells):
        if cellType[i] == SOLID_CELL:
            cellColor[3 * i] = 0.5
            cellColor[3 * i + 1] = 0.5
            cellColor[3 * i + 2] = 0.5
        elif cellType[i] == FLUID_CELL:
            d = particleDensity[i]
            if particleRestDensity > 0.0:
                d /= particleRestDensity
            # setSciColor_func(i, d, 0.0, 2.0, cellColor)

@ti.func
def setSciColor_func(cellNr, val, minVal, maxVal, cellColor):
    val = min(max(val, minVal), maxVal - 0.0001)
    d = maxVal - minVal
    val = 0.5 if d == 0.0 else (val - minVal) / d
    m = 0.25
    num = ti.floor(val / m, dtype=int)
    s = (val - num * m) / m

    if num == 0:
        r = 0.0
        g = s
        b = 1.0
    elif num == 1:
        r = 0.0
        g = 1.0
        b = 1.0 - s
    elif num == 2:
        r = s
        g = 1.0
        b = 0.0
    elif num == 3:
        r = 1.0
        g = 1.0 - s
        b = 0.0

    cellColor[3 * cellNr] = r
    cellColor[3 * cellNr + 1] = g
    cellColor[3 * cellNr + 2] = b

@ti.kernel
def handleParticleCollisions_kernel(numParticles:int,particlePos:ti.types.ndarray(),particleVel:ti.types.ndarray(),particleRadius:float,
                                    obstacleX:float,obstacleY:float,obstacleRadius:float,obstacleVelX:float, obstacleVelY:float, pInvSpacing:float, pNumX:int, pNumY:int):
    h = 1.0 / pInvSpacing
    r = particleRadius
    # or_ = obstacleRadius
    # or2 = or_ * or_
    minDist = obstacleRadius + r
    minDist2 = minDist * minDist

    minX = h + r
    maxX = (pNumX - 1) * h - r
    minY = h + r
    maxY = (pNumY - 1) * h - r

    for i in range(numParticles):
        x = particlePos[2 * i]
        y = particlePos[2 * i + 1]

        dx = x - obstacleX
        dy = y - obstacleY
        d2 = dx * dx + dy * dy

        # obstacle collision
        if d2 < minDist2:
            particleVel[2 * i] = obstacleVelX
            particleVel[2 * i + 1] = obstacleVelY

        # wall collisions
        if x < minX:
            x = minX
            particleVel[2 * i] = 0.0
        if x > maxX:
            x = maxX
            particleVel[2 * i] = 0.0
        if y < minY:
            y = minY
            particleVel[2 * i + 1] = 0.0
        if y > maxY:
            y = maxY
            particleVel[2 * i + 1] = 0.0

        particlePos[2 * i] = x
        particlePos[2 * i + 1] = y



@ti.kernel
def integrateParticles_kernel(numParticles:int, particleVel:ti.types.ndarray(), particlePos:ti.types.ndarray(), dt:float, gravity:float):
        for i in range(numParticles):
            particleVel[2 * i + 1] += dt * gravity
            particlePos[2 * i] += particleVel[2 * i] * dt
            particlePos[2 * i + 1] += particleVel[2 * i + 1] * dt
# ---------------------------------------------------------------------------- #
#                                  FLIP FLUID                                  #
# ---------------------------------------------------------------------------- #




class FlipFluid:
    def __init__(self, density, width, height, spacing, particleRadius, maxParticles:int) -> None:
        # fluid
        self.density = density
        self.fNumX = np.floor(width / spacing) + 1
        self.fNumY = np.floor(height / spacing) + 1
        self.fNumX = int(self.fNumX)
        self.fNumY = int(self.fNumY)
        self.h = max(width / self.fNumX, height / self.fNumY)
        self.fInvSpacing = 1.0 / self.h
        self.fNumCells = self.fNumX * self.fNumY
        self.fNumCells = int(self.fNumCells)

        self.u = np.zeros(self.fNumCells)
        self.v = np.zeros(self.fNumCells)
        self.du = np.zeros(self.fNumCells)
        self.dv = np.zeros(self.fNumCells)
        self.prevU = np.zeros(self.fNumCells)
        self.prevV = np.zeros(self.fNumCells)
        self.p = np.zeros(self.fNumCells)
        self.s = np.zeros(self.fNumCells)
        self.cellType = np.zeros(self.fNumCells)
        self.cellColor = np.zeros(3 * self.fNumCells)

        # particles

        self.maxParticles = maxParticles

        self.particlePos = np.zeros(2 * self.maxParticles)
        self.particleColor = np.zeros(3 * self.maxParticles)
        for i in range(self.maxParticles):
            self.particleColor[3 * i + 2] = 1.0

        self.particleVel = np.zeros(2 * self.maxParticles)
        self.particleDensity = np.zeros(self.fNumCells)
        self.particleRestDensity = 0.0

        self.particleRadius = particleRadius
        self.pInvSpacing = 1.0 / (2.2 * particleRadius)
        self.pNumX = np.floor(width * self.pInvSpacing) + 1
        self.pNumY = np.floor(height * self.pInvSpacing) + 1
        self.pNumX = int(self.pNumX)
        self.pNumY = int(self.pNumY)
        self.pNumCells = self.pNumX * self.pNumY

        self.numCellParticles = np.zeros(self.pNumCells, int)
        self.firstCellParticle = np.zeros(self.pNumCells + 1, int)
        self.cellParticleIds = np.zeros(self.maxParticles, int)

        self.numParticles = 0

    def integrateParticles(self, dt, gravity):
        integrateParticles_kernel(self.numParticles, self.particleVel, self.particlePos, dt, gravity)
        # for i in range(self.numParticles):
        #     self.particleVel[2 * i + 1] += dt * gravity
        #     self.particlePos[2 * i] += self.particleVel[2 * i] * dt
        #     self.particlePos[2 * i + 1] += self.particleVel[2 * i + 1] * dt

    def pushParticlesApart(self, numIters):
        colorDiffusionCoeff = 0.001

        # count particles per cell
        self.numCellParticles.fill(0)

        pushParticlesApart_countParticlesPerCell_kernel(self.numParticles, self.pNumX, self.pNumY,
                                                     self.pInvSpacing, self.particlePos, self.numCellParticles)
        # for i in range(self.numParticles):
        #     x = self.particlePos[2 * i]
        #     y = self.particlePos[2 * i + 1]

        #     xi = clamp(np.floor(x * self.pInvSpacing), 0, self.pNumX - 1)
        #     yi = clamp(np.floor(y * self.pInvSpacing), 0, self.pNumY - 1)
        #     xi = int(xi)
        #     yi = int(yi)
        #     cellNr = xi * self.pNumY + yi
        #     cellNr = int(cellNr)
        #     self.numCellParticles[cellNr] += 1


        # partial sums
        pushParticlesApart_partialSums_kernel(self.pNumCells, self.numCellParticles,self.firstCellParticle)
        # first = 0
        # for i in range(self.pNumCells):
        #     first += self.numCellParticles[i]
        #     self.firstCellParticle[i] = first
        # self.firstCellParticle[self.pNumCells] = first # guard


        # fill cell particle ids
        pushParticlesApart_fillCellParticleIds_kernel(self.numParticles, self.pNumX, self.pNumY,self.pInvSpacing,
                                                    self.particlePos, self.firstCellParticle,
                                                    self.cellParticleIds)
        # for i in range(self.numParticles):
        #     x = self.particlePos[2 * i]
        #     y = self.particlePos[2 * i + 1]

        #     xi = clamp(np.floor(x * self.pInvSpacing), 0, self.pNumX - 1)
        #     yi = clamp(np.floor(y * self.pInvSpacing), 0, self.pNumY - 1)
        #     xi = int(xi)
        #     yi = int(yi)
        #     cellNr = xi * self.pNumY + yi
        #     cellNr = int(cellNr)
        #     self.firstCellParticle[cellNr] -= 1
        #     self.cellParticleIds[self.firstCellParticle[cellNr]] = i

        # push particles apart
        pushParticlesApart_kernel(self.particleRadius, colorDiffusionCoeff, numIters,
                               self.numParticles, self.pNumX, self.pNumY, self.pInvSpacing,
                               self.particlePos, self.firstCellParticle,
                                 self.cellParticleIds, self.particleColor )
        # minDist = 2.0 * self.particleRadius
        # minDist2 = minDist * minDist
        # for iter in range(numIters):
        #     for i in range(self.numParticles):
        #         px = self.particlePos[2 * i]
        #         py = self.particlePos[2 * i + 1]

        #         pxi = clamp(np.floor(px * self.pInvSpacing), 0, self.pNumX - 1)
        #         pyi = clamp(np.floor(py * self.pInvSpacing), 0, self.pNumY - 1)
        #         x0 = max(pxi - 1, 0)
        #         y0 = max(pyi - 1, 0)
        #         x1 = min(pxi + 1, self.pNumX - 1)
        #         y1 = min(pyi + 1, self.pNumY - 1)
        #         x0 = int(x0)
        #         y0 = int(y0)
        #         x1 = int(x1)
        #         y1 = int(y1)

        #         for xi in range(x0, x1 + 1):
        #             for yi in range(y0, y1+1):
        #                 cellNr = xi * self.pNumY + yi
        #                 first = self.firstCellParticle[cellNr]
        #                 last = self.firstCellParticle[cellNr + 1]
        #                 for j in range(first, last):
        #                     id = self.cellParticleIds[j]
        #                     if id == i:
        #                         continue
        #                     qx = self.particlePos[2 * id]
        #                     qy = self.particlePos[2 * id + 1]

        #                     dx = qx - px
        #                     dy = qy - py
        #                     d2 = dx * dx + dy * dy

        #                     if (d2 > minDist2 or d2 == 0.0):
        #                         continue

        #                     d = np.sqrt(d2)
        #                     s = 0.5 * (minDist - d) / d
        #                     dx *= s
        #                     dy *= s
        #                     self.particlePos[2 * i] -= dx
        #                     self.particlePos[2 * i + 1] -= dy
        #                     self.particlePos[2 * id] += dx
        #                     self.particlePos[2 * id + 1] += dy

        #                     # diffuse colors
        #                     for k in range(3):
        #                         color0 = self.particleColor[3 * i + k]
        #                         color1 = self.particleColor[3 * id + k]
        #                         color = 0.5 * (color0 + color1)
        #                         self.particleColor[3 * i + k] = color0 + colorDiffusionCoeff * (color - color0)
        #                         self.particleColor[3 * id + k] = color1 + colorDiffusionCoeff * (color - color1)

    def handleParticleCollisions(self, obstacleX, obstacleY, obstacleRadius):

        handleParticleCollisions_kernel(self.numParticles,self.particlePos,self.particleVel,self.particleRadius,
                                    obstacleX,obstacleY,obstacleRadius,scene.obstacleVelX, scene.obstacleVelY, self.pInvSpacing, self.pNumX, self.pNumY)
        # h = 1.0 / self.pInvSpacing
        # r = self.particleRadius
        # # or_ = obstacleRadius
        # # or2 = or_ * or_
        # minDist = obstacleRadius + r
        # minDist2 = minDist * minDist

        # minX = h + r
        # maxX = (self.pNumX - 1) * h - r
        # minY = h + r
        # maxY = (self.pNumY - 1) * h - r

        # for i in range(self.numParticles):
        #     x = self.particlePos[2 * i]
        #     y = self.particlePos[2 * i + 1]

        #     dx = x - obstacleX
        #     dy = y - obstacleY
        #     d2 = dx * dx + dy * dy

        #     # obstacle collision
        #     if d2 < minDist2:
        #         self.particleVel[2 * i] = scene.obstacleVelX
        #         self.particleVel[2 * i + 1] = scene.obstacleVelY

        #     # wall collisions
        #     if x < minX:
        #         x = minX
        #         self.particleVel[2 * i] = 0.0
        #     if x > maxX:
        #         x = maxX
        #         self.particleVel[2 * i] = 0.0
        #     if y < minY:
        #         y = minY
        #         self.particleVel[2 * i + 1] = 0.0
        #     if y > maxY:
        #         y = maxY
        #         self.particleVel[2 * i + 1] = 0.0

        #     self.particlePos[2 * i] = x
        #     self.particlePos[2 * i + 1] = y

    def transferVelocities(self, toGrid, flipRatio=0.0):
        n = self.fNumY
        h = self.h
        h1 = self.fInvSpacing
        h2 = 0.5 * h

        if toGrid:
            self.prevU= (self.u).copy()
            self.prevV = (self.v).copy()

            self.du.fill(0.0)
            self.dv.fill(0.0)
            self.u.fill(0.0)
            self.v.fill(0.0)

            transferVelocities_setCellType_kernel(n, self.numParticles, self.fNumX, self.fNumY, h1, self.cellType, self.fNumCells, self.particlePos, self.s)
            # for i in range(self.fNumCells):
            #     self.cellType[i] = SOLID_CELL if self.s[i] == 0.0 else AIR_CELL

            # for i in range(self.numParticles):
            #     x = self.particlePos[2 * i]
            #     y = self.particlePos[2 * i + 1]
            #     xi = clamp(np.floor(x * h1), 0, self.fNumX - 1)
            #     yi = clamp(np.floor(y * h1), 0, self.fNumY - 1)
            #     xi = int(xi)
            #     yi = int(yi)
            #     cellNr = xi * n + yi
            #     cellNr = int(cellNr)
            #     if self.cellType[cellNr] == AIR_CELL:
            #         self.cellType[cellNr] = FLUID_CELL

        
        for component in range(2):
            dx = 0.0 if component == 0 else h2
            dy = h2 if component == 0 else 0.0

            f = self.u if component == 0 else self.v
            prevF = self.prevU if component == 0 else self.prevV
            d = self.du if component == 0 else self.dv

            transferVelocities_kernel(n, self.numParticles,h, h1, dx, dy, self.fNumX, self.fNumY,
                                self.particlePos, self.particleVel, 
                                  f, toGrid, component, self.cellType,
                                  prevF, d, flipRatio )
            # for i in range(self.numParticles):
            #     x = self.particlePos[2 * i]
            #     y = self.particlePos[2 * i + 1]

            #     x = clamp(x, h, (self.fNumX - 1) * h)
            #     y = clamp(y, h, (self.fNumY - 1) * h)

            #     x0 = min(np.floor((x - dx) * h1), self.fNumX - 2)
            #     tx = (x - dx) - x0 * h
            #     x1 = min(x0 + 1, self.fNumX - 2)

            #     y0 = min(np.floor((y - dy) * h1), self.fNumY - 2)
            #     ty = ((y - dy) - y0*h) * h1
            #     y1 = min(y0 + 1, self.fNumY - 2)

            #     sx = 1.0 - tx
            #     sy = 1.0 - ty

            #     d0 = sx * sy
            #     d1 = tx * sy
            #     d2 = tx * ty
            #     d3 = sx * ty

            #     nr0 = x0 * n + y0
            #     nr1 = x1 * n + y0
            #     nr2 = x1 * n + y1
            #     nr3 = x0 * n + y1

            #     nr0 = int(nr0)
            #     nr1 = int(nr1)
            #     nr2 = int(nr2)
            #     nr3 = int(nr3)

            #     if toGrid:
            #         pv = self.particleVel[2 * i + component]
            #         f[nr0] += pv * d0
            #         d[nr0] += d0
            #         f[nr1] += pv * d1
            #         d[nr1] += d1
            #         f[nr2] += pv * d2
            #         d[nr2] += d2
            #         f[nr3] += pv * d3
            #         d[nr3] += d3

            #     else:
            #         offset = n if component == 0 else 1
            #         valid0 = 1.0 if self.cellType[nr0] != AIR_CELL or self.cellType[nr0 - offset] != AIR_CELL else 0.0
            #         valid1 = 1.0 if self.cellType[nr1] != AIR_CELL or self.cellType[nr1 - offset] != AIR_CELL else 0.0
            #         valid2 = 1.0 if self.cellType[nr2] != AIR_CELL or self.cellType[nr2 - offset] != AIR_CELL else 0.0
            #         valid3 = 1.0 if self.cellType[nr3] != AIR_CELL or self.cellType[nr3 - offset] != AIR_CELL else 0.0

            #         v = self.particleVel[2 * i + component]
            #         d = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3

            #         if d > 0.0:
            #             picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / d
            #             corr = (valid0 * d0 * (f[nr0] - prevF[nr0]) + valid1 * d1 * (f[nr1] - prevF[nr1]) + valid2 * d2 * (f[nr2] - prevF[nr2]) + valid3 * d3 * (f[nr3] - prevF[nr3])) / d
            #             flipV = v + corr
            #             self.particleVel[2 * i + component] = (1.0 - flipRatio) * picV + flipRatio * flipV

            if toGrid:
                # for i in range(len(f)):
                #     if d[i] > 0.0:
                #         f[i] /= d[i]

                # restore solid cells
                transferVelocities_restoreSolidCells_kernel(self.fNumX, self.fNumY, n, self.cellType, self.u, self.v, self.prevU, self.prevV,f, d)

                # for i in range(self.fNumX):
                #     for j in range(self.fNumY):
                #         solid = self.cellType[i * n + j] == SOLID_CELL
                #         if solid or (i > 0 and self.cellType[(i - 1) * n + j] == SOLID_CELL):
                #             self.u[i * n + j] = self.prevU[i * n + j]
                #         if solid or (j > 0 and self.cellType[i * n + j - 1] == SOLID_CELL):
                #             self.v[i * n + j] = self.prevV[i * n + j]


    def updateParticleDensity(self):
        n = self.fNumY
        h = self.h
        h1 = self.fInvSpacing
        h2 = 0.5 * h

        d = self.particleDensity

        d.fill(0.0)

        updateParticleDensity_kernel(self.numParticles,self.particlePos,
                                 h,h1,h2,self.fNumX,self.fNumY,d, n)
        # for i in range(self.numParticles):
        #     x = self.particlePos[2 * i]
        #     y = self.particlePos[2 * i + 1]

        #     x = clamp(x, h, (self.fNumX - 1) * h)
        #     y = clamp(y, h, (self.fNumY - 1) * h)

        #     x0 = np.floor((x - h2) * h1)
        #     tx = (x - h2) - x0 * h
        #     x1 = min(x0 + 1, self.fNumX - 2)

        #     y0 = np.floor((y - h2) * h1)
        #     ty = (y - h2) - y0 * h
        #     y1 = min(y0 + 1, self.fNumY - 2)

        #     sx = 1.0 - tx
        #     sy = 1.0 - ty

        #     x0 = int(x0)
        #     x1 = int(x1)
        #     y0 = int(y0)
        #     y1 = int(y1)

        #     if x0 < self.fNumX and y0 < self.fNumY:
        #         d[x0 * n + y0] += sx * sy
        #     if x1 < self.fNumX and y0 < self.fNumY:
        #         d[x1 * n + y0] += tx * sy
        #     if x1 < self.fNumX and y1 < self.fNumY:
        #         d[x1 * n + y1] += tx * ty
        #     if x0 < self.fNumX and y1 < self.fNumY:
        #         d[x0 * n + y1] += sx * ty


        if self.particleRestDensity == 0.0:
            sum = 0.0
            numFluidCells = 0

            for i in range(self.fNumCells):
                if self.cellType[i] == FLUID_CELL:
                    sum += d[i]
                    numFluidCells += 1

            if numFluidCells > 0:
                self.particleRestDensity = sum / numFluidCells





    def solveIncompressibility(self, numIters, dt, overRelaxation, compensateDrift = True):
        self.p.fill(0.0)
        self.prevU = (self.u).copy()
        self.prevV = (self.v).copy()

        n = self.fNumY
        cp = self.density * self.h / dt

        # for i in range(self.fNumCells):
        #     u = self.u[i]
        #     v = self.v[i]

        for iter in range(numIters):
            solveIncompressibility_kernel(n, cp, self.fNumX, self.fNumY, self.fInvSpacing, self.u, self.v, self.p, self.s, self.cellType, self.particleRestDensity, self.particleDensity, overRelaxation, compensateDrift)
            # for i in range(1, self.fNumX - 1):
            #     for j in range(1, self.fNumY - 1):
            #         if self.cellType[i * n + j] != FLUID_CELL:
            #             continue

            #         center = i * n + j
            #         left = (i - 1) * n + j
            #         right = (i + 1) * n + j
            #         bottom = i * n + j - 1
            #         top = i * n + j + 1

            #         sc = self.s[center]
            #         sx0 = self.s[left]
            #         sx1 = self.s[right]
            #         sy0 = self.s[bottom]
            #         sy1 = self.s[top]
            #         sc = (sx0 + sx1 + sy0 + sy1)

            #         if sc == 0.0:
            #             continue

            #         div = self.u[right] - self.u[center] + self.v[top] - self.v[center]

            #         if self.particleRestDensity > 0.0 and compensateDrift:
            #             k = 1.0
            #             compression = self.particleDensity[i * n + j] - self.particleRestDensity
            #             if compression > 0.0:
            #                 div = div - k * compression

            #         p = -div / sc
            #         p *= overRelaxation
            #         self.p[center] += cp * p

            #         self.u[center] -= sx0 * p
            #         self.u[right] += sx1 * p
            #         self.v[center] -= sy0 * p
            #         self.v[top] += sy1 * p
                    
    def  updateParticleColors(self):
        updateParticleColors_kernel(self.numParticles,self.particlePos,self.particleRestDensity,self.particleDensity,
                                    self.fNumX,self.fNumY,self.particleColor, self.fInvSpacing, self.h)
        # h1 = self.fInvSpacing
		
        # for i in range(self.numParticles):
        #     s = 0.01

        #     self.particleColor[3 * i] = clamp(self.particleColor[3 * i] - s, 0.0, 1.0)
        #     self.particleColor[3 * i + 1] = clamp(self.particleColor[3 * i + 1] - s, 0.0, 1.0)
        #     self.particleColor[3 * i + 2] = clamp(self.particleColor[3 * i + 2] + s, 0.0, 1.0)

        #     x = self.particlePos[2 * i]
        #     y = self.particlePos[2 * i + 1]
        #     xi = clamp(np.floor((x - 0.5 * self.h) * h1), 0, self.fNumX - 1)
        #     yi = clamp(np.floor((y - 0.5 * self.h) * h1), 0, self.fNumY - 1)
        #     cellNr = xi * self.fNumY + yi
        #     cellNr = int(cellNr)

        #     d0 = self.particleRestDensity

        #     if d0 > 0.0:
        #         relDensity = self.particleDensity[cellNr] / d0
        #         if relDensity < 0.7:
        #             s = 0.8
        #             self.particleColor[3 * i] = s
        #             self.particleColor[3 * i + 1] = s
        #             self.particleColor[3 * i + 2] = 1.0

    
    def setSciColor(self, cellNr, val, minVal, maxVal):
        val = min(max(val, minVal), maxVal - 0.0001)
        d = maxVal - minVal
        val = 0.5 if d == 0.0 else (val - minVal) / d
        m = 0.25
        num = np.floor(val / m)
        s = (val - num * m) / m

        if num == 0:
            r = 0.0
            g = s
            b = 1.0
        elif num == 1:
            r = 0.0
            g = 1.0
            b = 1.0 - s
        elif num == 2:
            r = s
            g = 1.0
            b = 0.0
        elif num == 3:
            r = 1.0
            g = 1.0 - s
            b = 0.0

        self.cellColor[3 * cellNr] = r
        self.cellColor[3 * cellNr + 1] = g
        self.cellColor[3 * cellNr + 2] = b



    
    def updateCellColors(self):
        self.cellColor.fill(0.0)

        updateCellColors_kernel(self.cellColor,self.cellType,self.particleDensity,self.particleRestDensity,self.fNumCells)

        # for i in range(self.fNumCells):
        #     if self.cellType[i] == SOLID_CELL:
        #         self.cellColor[3 * i] = 0.5
        #         self.cellColor[3 * i + 1] = 0.5
        #         self.cellColor[3 * i + 2] = 0.5
        #     elif self.cellType[i] == FLUID_CELL:
        #         d = self.particleDensity[i]
        #         if self.particleRestDensity > 0.0:
        #             d /= self.particleRestDensity
        #         self.setSciColor(i, d, 0.0, 2.0)



    def simulate(self, dt, gravity, flipRatio, numPressureIters, numParticleIters, overRelaxation, compensateDrift, separateParticles, obstacleX, abstacleY, obstacleRadius):
        numSubSteps = 1
        sdt = dt / numSubSteps

        for step in range(numSubSteps):
            self.integrateParticles(sdt, gravity)
            if separateParticles:
                self.pushParticlesApart(numParticleIters)
            self.handleParticleCollisions(obstacleX, abstacleY, obstacleRadius)
            self.transferVelocities(True)
            self.updateParticleDensity()
            self.solveIncompressibility(numPressureIters, sdt, overRelaxation, compensateDrift)
            self.transferVelocities(False, flipRatio)

        # self.updateParticleColors()
        # self.updateCellColors()
    



def setObstacle(x, y, reset):
    vx = 0.0
    vy = 0.0

    if not reset:
        vx = (x - scene.obstacleX) / scene.dt
        vy = (y - scene.obstacleY) / scene.dt

    scene.obstacleX = x
    scene.obstacleY = y
    r = scene.obstacleRadius
    f = scene.fluid

    # n = f.fnumY
    # cd = np.sqrt(2) * f.h

    # for i in range(1, f.fnumX - 2):
    #     for j in range(1, f.fnumY - 2):
    #         f.s[i * n + j] = 1.0

    #         dx = (i + 0.5) * f.h - x
    #         dy = (j + 0.5) * f.h - y

    #         if dx * dx + dy * dy < r * r:
    #             f.s[i * n + j] = 0.0
    #             f.u[i * n + j] = vx
    #             f.u[(i + 1) * n + j] = vx
    #             f.v[i * n + j] = vy
    #             f.v[i * n + j + 1] = vy
    
    scene.showObstacle = True
    scene.obstacleVelX = vx
    scene.obstacleVelY = vy


def setupScene():
    scene.obstacleRadius = 0.15
    scene.overRelaxation = 1.9

    scene.dt = 1.0 / 60.0
    scene.numPressureIters = 50

    scene.numParticleIters = 2

    res = 100

    tankHeight = 1.0 * simHeight 
    tankWidth = 1.0 * simWidth 
    h = tankHeight / res
    density = 1000.0

    relWaterHeight = 0.8 
    relWaterWidth = 0.6 

    # dam break

    # compute number of particles

    r = 0.3 * h  # particle radius w.r.t. cell size
    dx = 2.0 * r
    dy = np.sqrt(3.0) / 2.0 * dx

    numX = np.floor((relWaterWidth * tankWidth - 2.0 * h - 2.0 * r) / dx).astype(int)
    numY = np.floor((relWaterHeight * tankHeight - 2.0 * h - 2.0 * r) / dy).astype(int)
    maxParticles = numX * numY

    # create fluid

    f = scene.fluid = FlipFluid(density, tankWidth, tankHeight, h, r, maxParticles)

    # create particles
    f.numParticles = numX * numY

    p = 0
    for i in range(numX):
        for j in range(numY):
            f.particlePos[p] = h + r + dx * i + (0.0 if j % 2 == 0  else r)
            f.particlePos[p + 1] = h + r + dy * j
            p += 2

    # setup grid cells for tank

    n = f.fNumY

    for i in range(f.fNumX):
        for j in range(f.fNumY):
            s = 1.0 # fluid
            if i == 0 or i == f.fNumX - 1 or j == 0:
                s = 0.0
            f.s[i * n + j] = s

    setObstacle(3.0, 2.0, True)


def transferParticlePos():
    pos = scene.fluid.particlePos
    posx = pos[0::2]
    posy = pos[1::2]
    particlePos = np.zeros((scene.fluid.numParticles, 2))
    particlePos[:,0] = posx
    particlePos[:,1] = posy
    #normalize to 0 to 1
    particlePos[:,0] /= simWidth
    particlePos[:,1] /= simHeight
    return particlePos

def main():
    setupScene()

    scene.paused = False

    gui = ti.GUI("FLIP",res=(640,640))
    gui.background_color = 0xFFFFFF
    while gui.running:
        if not scene.paused :
            scene.fluid.simulate(
                scene.dt, scene.gravity, scene.flipRatio, scene.numPressureIters, scene.numParticleIters, 
                scene.overRelaxation, scene.compensateDrift, scene.separateParticles,
                scene.obstacleX, scene.obstacleY, scene.obstacleRadius)
            scene.frameNr += 1

        particlePos = transferParticlePos()
        gui.clear(0xFFFFFF)
        gui.circles(pos=particlePos, radius=1, color=0x0000FF)
        gui.show()

        # filename = f'img/{scene.frameNr:04d}.png'
        # gui.show(filename)
        # if scene.frameNr == 200:
        #     exit()


if __name__ == "__main__":
    main()
