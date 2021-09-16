import vtk
# https://pyscience.wordpress.com/2014/11/16/volume-rendering-with-python-and-vtk/
# https://kitware.github.io/vtk-examples/site/Python/Tutorial/Tutorial_Step1/

colors = vtk.vtkNamedColors()

reader = vtk.vtkNrrdReader()
reader.SetFileName(r'F:\Xiaotang\uCT\P0_20201003\screen_Man3\12hr_P0_brain_recon.nrrd')

alphaFunc = vtk.vtkPiecewiseFunction()
alphaFunc.AddPoint(0, 0.0)
alphaFunc.AddPoint(50, 0.0)
alphaFunc.AddPoint(128, 0.3)
alphaFunc.AddPoint(191, 0.4)
alphaFunc.AddPoint(255, 0.6)

colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(0, 0.75, 0.75, 0.75)
colorFunc.AddRGBPoint(128, 0.75, 0.75, 0.75)
colorFunc.AddRGBPoint(255, 0.5, 0.5, 0.5)

volProp = vtk.vtkVolumeProperty()
volProp.SetColor(colorFunc)
volProp.ShadeOn()
volProp.SetAmbient(0.25)
volProp.SetDiffuse(0.75)
volProp.SetSpecular(0)
volProp.SetScalarOpacity(alphaFunc)
volProp.SetInterpolationTypeToLinear()

# volMapper = vtk.vtkFixedPointVolumeRayCastMapper()
volMapper = vtk.vtkSmartVolumeMapper()
volMapper.SetInputConnection(reader.GetOutputPort())

vol = vtk.vtkVolume()
vol.SetMapper(volMapper)
vol.SetProperty(volProp)

renderer = vtk.vtkRenderer()
renderer.AddVolume(vol)
renderer.SetBackground(colors.GetColor3d('White'))
renderer.ResetCamera()
renderer.GetActiveCamera().Azimuth(-90)
renderer.GetActiveCamera().Elevation(-90)
# renderer.GetActiveCamera().Roll(180)
renderer.GetActiveCamera().Zoom(1.5)

renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
# renderWin.SetSize(1920 , 1080)
renderWin.SetSize(1280 , 720)
renderWin.Render()

# win2img = vtk.vtkWindowToImageFilter()
# win2img.SetInput(renderWin)
# win2img.Update()
# writer = vtk.vtkPNGWriter()
# writer.SetFileName(r"F:\Xiaotang\uCT\P0_20201003\screen_Man3\view0.png")
# writer.SetInputData(win2img.GetOutput())
# writer.Write()

renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)
renderInteractor.Initialize()
renderInteractor.Start()
