# Analysing-Medical-Transcipts-using-Project-Debater

# Project-Debater


Project Debater is the first AI system that can debate humans on complex topics. Project Debater digests massive texts, constructs a well-structured speech on a given topic, delivers it with clarity and purpose, and rebuts its opponent. Eventually, Project Debater will help people reason by providing compelling, evidence-based arguments and limiting the influence of emotion, bias, or ambiguity.


In this workshop you will get an insight on how to use Project Debater to analyse and derive insights from medical transcipts. 

## Prerequisites

You can receive credentials for the Project Debater APIs [early-access-program site](https://early-access-program.debater.res.ibm.com) by sending a request to `project.debater@il.ibm.com`

The Early Access Program is free for academic use, and is available for trial and licensing for commercial use:
- https://early-access-program.debater.res.ibm.com/academic_use.html
- https://early-access-program.debater.res.ibm.com/business_use


Download and install the Python SDK according to the instructions in:
https://early-access-program.debater.res.ibm.com/download_sdk.html



## Getting Started with Jupyter Notebooks

Jupyter notebooks are an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text. 

In this workshop we will use IBM Watson Studio to run a notebook. For this you will need an IBM Cloud account. The following steps will show you how sign up and get started. When you have the notebook up and running we will go through the notebook. 

## IBM Cloud

- [Sign up](cloud.ibm.com) for an IBM Cloud account

- When you are signed up click `Create Resource` at the top of the Resources page. You can find the resources under the hamburger menu at the top left:

 ![](https://github.com/IBMDeveloperUK/python-geopandas-workshop/blob/master/images/Create_resource.png)
 
- Search for Watson Studio and click on the tile:

![](https://github.com/IBMDeveloperUK/jupyter-notebooks-101/blob/master/images/studio.png)
- Select the Lite plan and click `Create`.
- Go back to the Resources list and click on your Watson Studio service and then click `Get Started`. 

![](https://github.com/IBMDeveloperUK/jupyter-notebooks-101/blob/master/images/launch.png)

## IBM Watson Studio

### 1. Create a new Project

- You should now be in Watson Studio.
- 
- Click on the Projects option to create a New project. 

![](https://github.com/YaminiRao/Data-Visualisation-with-Python/blob/master/Images/Watson_Studio.png)

- Select an Object Storage from the drop-down menu or create a new one for free. This is used to store the notebooks and data. **Do not forget to click refresh when returning to the Project page.**

![](https://github.com/IBMDeveloperUK/Machine-Learning-Models-with-AUTO-AI/blob/master/Images/COS.png)

- click `Create`.  


## 4. Load and run a notebook

-  Add a new notebook. Click `Add to project` and choose `Notebook`:

![](https://github.com/IBMDeveloperUK/python-geopandas-workshop/blob/master/images/notebook.png)

- Choose new notebook `From URL`. Give your notebook a name and copy the URL `https://github.com/IBMDeveloperUK/Classification-Models-using-Python-and-Scikit-Learn/blob/main/Notebook/Classification_models.ipynb`
- Select the custom runtime enviroment that you created and click `Create Notebook`. 
-  The notebook will load. 
 
<b> You are now ready to follow along with the workshop in the notebook! </b>
