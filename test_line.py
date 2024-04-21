import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

class ColoredLines(pg.GraphicsObject):
    def __init__(self, points, colors):
        super().__init__()
        self.points = points
        self.colors = colors
        self.generatePicture()
    
    def generatePicture(self):
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        pen = pg.functions.mkPen()

        for idx in range(len(self.points) - 1):
            pen.setColor(self.colors[idx])
            painter.setPen(pen)
            painter.drawLine(self.points[idx], self.points[idx+1])

        painter.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

xxx = np.linspace(0, 2, 1000)
xxcolor = np.concatenate((np.linspace(0, 1, 500), np.linspace(1, 0, 500)))

yyy = 2.0*np.sin(2*np.pi*1.0*xxx)
points = [QtCore.QPointF(*xy.tolist()) for xy in np.column_stack((xxx, yyy))]
# cmap = pg.colormap.get("CET-C1")
# lut = cmap.getLookupTable(nPts=100, mode="qcolor")
# colors = [lut[idx % len(lut)] for idx in range(len(xxx)-1)]

# map xxcolor to a color
def map_color(xxcolor):
    cmap = pg.colormap.get("viridis")
    lut = cmap.getLookupTable(nPts=100, mode="qcolor")
    # map values in xxcolor to integer between 0 and 99
    min = np.min(xxcolor)
    max = np.max(xxcolor)
    xxcolor = (xxcolor - min) / (max - min) * 99
    xxcolor = xxcolor.astype(int)
    colors = [lut[idx] for idx in xxcolor]
    return colors

colors = map_color(xxcolor)

pg.mkQApp()
plt = pg.PlotWidget()
plt.show()
item = ColoredLines(points, colors)
plt.addItem(item)
pg.exec()