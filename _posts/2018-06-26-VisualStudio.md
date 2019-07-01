---
title:  "IDE setting for Unreal"
categories: post
mathjax: true
---
- I am putting few good tips for Visual Studio (VS) setting to work with Unreal engine. 

   - [Setting up VS](https://docs.unrealengine.com/en-US/Programming/Development/VisualStudioSetup/index.html)
   - [Unreal VS extension](https://docs.unrealengine.com/en-US/Programming/Development/VisualStudioSetup/UnrealVS/index.html)
   - [Visual Assist- this is a paid s/w helps development](https://www.wholetomato.com/)
   - Turn off squggles @ VS options->Text Editor->C/C++>Advanced->Intellisense->Disable Squiggles "True"
   - VS code is also working with Unreal (4.18 or higher). 
   - How to set up VS code to use VS compiler is, 
     - Install [C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) from VS market place. 
     - Make sure that you had installed [VS build tool](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017) 
     - Go to File, Settings, type shell, then look for Edit in settings.json: 
   ![settings.json](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/settings_VS_Code.png)
     - Look for where VsDevCmd.bat file is located in your local path, then type as shown below:
   ![VSDevCmd.bat](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/VsDevCmd.png) 
     - You can also open the code in terminal, directly:
   ![OpenInTerminal](https://github.com/SeokLeeUS/seokleeus.github.io/raw/master/_images/_VS/OpenInTerminal.png)   
     
   
   
