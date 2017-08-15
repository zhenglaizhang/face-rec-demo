import wx
import cv2


class viewWindow(wx.Frame):
    def __init__(self, parent, title="View Window"):
        # super(viewWindow,self).__init__(parent)
        wx.Frame.__init__(self, parent)

        self.imgSizer = (480, 360)
        self.pnl = wx.Panel(self)
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.image = wx.EmptyImage(self.imgSizer[0], self.imgSizer[1])
        self.imageBit = wx.BitmapFromImage(self.image)
        self.staticBit = wx.StaticBitmap(self.pnl, wx.ID_ANY, self.imageBit)

        self.vbox.Add(self.staticBit)

        self.capture = cv2.VideoCapture(0)
        ret, self.frame = self.capture.read()
        if ret:
            self.height, self.width = self.frame.shape[:2]
            self.bmp = wx.BitmapFromBuffer(self.width, self.height, self.frame)

            self.timex = wx.Timer(self)
            self.timex.Start(1000. / 24)
            self.Bind(wx.EVT_TIMER, self.redraw)
            self.SetSize(self.imgSizer)
        else:
            print("Error no webcam image")
        self.pnl.SetSizer(self.vbox)
        self.vbox.Fit(self)
        self.Show()

    def redraw(self, e):
        ret, self.frame = self.capture.read()
        if ret:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.bmp.CopyFromBuffer(self.frame)
            self.staticBit.SetBitmap(self.bmp)
            self.Refresh()


def main():
    app = wx.PySimpleApp()
    frame = viewWindow(None)
    frame.Center()
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
