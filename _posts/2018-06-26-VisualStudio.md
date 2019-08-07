---
title:  "IDE setting for Unreal"
categories: post
mathjax: true
---
- I am putting few good tips for Visual Studio (VS) setting to work with Unreal engine. 

   - Ctrl+k+c: comment the highlighted lines 
   - Ctrl+k+u: uncomment the highlighted lines
   - You can convert a line of code to a function by using this [method](https://www.udemy.com/unrealcourse/learn/lecture/4722784#questions/3494386). 
   - [Visual studio layout tip](https://docs.microsoft.com/en-us/visualstudio/ide/customizing-window-layouts-in-visual-studio?view=vs-2019)
   - [Setting up VS](https://docs.unrealengine.com/en-US/Programming/Development/VisualStudioSetup/index.html)
   - [Unreal VS extension](https://docs.unrealengine.com/en-US/Programming/Development/VisualStudioSetup/UnrealVS/index.html)
   - [Visual Assist- this is a paid s/w helps development](https://www.wholetomato.com/)
   - Turn off squggles @ VS options->Text Editor->C/C++>Advanced->Intellisense->Disable Squiggles "True"
   - VS code is also working with Unreal (4.18 or higher). 
   - How to set up VS code and some tips: 
     - Install [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) from VS market place. 
     - Make sure that you had installed [VS build tool](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017) 
     - Go to File, Settings, type shell, then look for Edit in settings.json: 
   ![settings.json](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/settings_VS_Code.png)
     - Look for where VsDevCmd.bat file is located in your local path, then type as shown below:
   ![VSDevCmd.bat](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/VsDevCmd.png) 
     - You can also open the code in terminal, directly:
   ![OpenInTerminal](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/OpenInTerminal.png)   
     - How to close multiple command terminals:
   ![Dustbin](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/Dustbin.png)   
     - Type "cl" command in terminal to see if the compiler sets correctly:
   ![cl_cmd](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/cl_command.png)  
     - How to set up source code editor @ Unreal engine:
   ![SourceCodeEditor](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/source_code_editor3.png)
   
   
   
   
   
