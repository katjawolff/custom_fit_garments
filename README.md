# custom_fit_garments
Creating and adjusting a 3D garment shape to a variety of 3D scanned poses


## Installation 
This code was tested on Mac, Ubuntu with Clion and Windows with Visual Studio 2017 and 2019. 
Use Sourcetree to clone https://github.com/katjawolff/custom_fit_garments.git
Check option 'Recurse Submodules'.
On Windows, open the project folder with Visual Studio (tested with versions 2017 or 2019). It will automatically generate the project from the cmake file. Compile garmentshape.exe as x64-Release.
Cholmod can be optionally installed for faster run times and the corrsponding path needs to be manually set in the cmake file. 

## Running 
The project comes with an example avatar. To start a simple project, use the button 'Load avatars from folder', navigate to /data/example_avatar/ and click any file to load the whole set of avatars. A window with a list of available avatars opens. Choose one to start with and you can start to create garments with the available tools. 

Note that this is research code and bugs will appear when the tools are not used in the right order. 

Follow this workflow, to create a pregnancy dress:

	- load avatars from folder
	- select the no-belly T-pose
	- create boundaries to outline a shirt 
	- set offset to 0.008
	- create garment from boundaries (scale 0.4)
	- extend into dress with Duplicate Boundary
	- start
	- select belly T-pose and wait until this pose is reached
	- use key S to pause any time and the Start putton to continue
