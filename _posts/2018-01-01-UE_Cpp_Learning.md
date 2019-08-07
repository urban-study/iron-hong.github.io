
This is a minimum to learn in order to carry Unreal/C++ related project work 
(work in progress to add reference materials, demo codes, and reference):

- Visual Studio and Visual Code IDEs (Integrated Development Environment) set up, some additional steps to expand features, and its associated useful shortcut keys
- User Interface (UI) set up- i.e. HUD
- Debugging and its tip
- Object Oriented Programming (OOP)
- Class inheritance 
- Handful tips to operate Unreal UI and BluePrint(BP) interfaces 
- Access modifier (private/public/protected)
- Architect and refactor the code 
- Polymorphism 
- Handling static mesh (LOD, Lighting, Shadowing) ,scene, actor,collision
- Version control (Unreal/VS/github)
- C++ syntax @ low and high level (pointers and references, type casting, macro,  etc.) and Unreal coding standard
- Handling camera control 
- Reference materials (C++ Unreal API reference) 
- Unreal naming convention 

Here are learning materials:

1.	Unreal BluePrint (BP) has a built in feature to shoot out box shape ray tracing for debugging purpose. This is the official material from Epic. A video demonstrates how to set up Box trace by channel. 

2.	In order to apply a function globally available to other BPs globally, there’s a solution called BP function library .

3.	 In order to make parameters available on other BPs globally , there’s a solution to make the variables available via a BP actor. 

4.	This demo walks you through how to set ray tracing using BP. 

5.	MathWorksSimulation plugin allows you to transfer ray tracing information to Simulink for co-simulation.  Check the compatible Unreal version for plugin installation. 

6.	There’s a radar HUD demo which you might find some insights for different implementation. 

7.	For ray tracing, collision type setting is important for other players. Please see this detailed information.

8.	 This reference explains how to make C++ class accessible by BP. 

9.	This is the form when you want to report a UE bug to Epic games. Before submitting a bug report, you can search here if your issue is already reported by someone else. 

10.	In order to utilize input command via keyboard or mouse to move the player, consider this material.

11.	 This material explains about lighting property setting. 

12.	When you want to know about the class inheritance hierarchy, please check Unreal official documentation. For instance,  you will see the inheritance at the top header file. 

13.	This does explain array container in UE. You can find some insight how the array works in BP from this documentation as well. 

14.	This describes how to remove unwanted C++ class from unreal project. 

15.	When you build code from VS, check this out first. 

16.	This describes directory structure. The bottom line is, you can remove build/binary/intermediate/saved folders when you recreate visual studio code file. 

17.	This describes how to control viewport. 

18.	This does describes UE gameplay framework. 
