import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen():

    hwin = win32gui.GetDesktopWindow()

    hwnd = win32gui.FindWindow(None, "Super Hexagon")

    rect = win32gui.GetWindowRect(hwnd)
    left = rect[0]
    top = rect[1]
    width = rect[2] - left
    height = rect[3] - top

    left += 8 - 1
    top += 32 - 1

    width -= 8 + 8 - 2
    height -= 32 + 8 - 1

    # print("Window %s:" % win32gui.GetWindowText(hwnd))
    # print("\tLocation: (%d, %d)" % (top, left))
    # print("\t    Size: (%d, %d)" % (width, height))

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    # img = np.fromstring(signedIntsArray, dtype='uint8')
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    # return img
