### Sujet
Le sujet qui accompagne ce TP est dans lab\_text.md


### Environnement de travail
Pour faire tourner ce TP, beaucoup de librairies sont indispensables, et notamment la robotics-toolbox de P. Corke (P.I. Corke : https://github.com/petercorke/robotics-toolbox-python/wiki). Afin de bénéficier d'un environement de travail complet, je vous suggère de procéder de la sorte : 

- Installer Docker.

- Aller dans le dossier docker_install et vérifier bien la présence d'un dossier volume\_tp. 

- Ecrire :

    sudo docker pull waelbachta/ubuntu_ssh_labs:latest

Cela vous permet de télécharger une image du docker prête à l'emploi. Vous allez devoir utiliser votre mot de passe personnel.

    sudo docker build -t waelbachta/ubuntu_ssh_labs:latest .

 Cela vous permettra de créer un conteneur basé sur l'image téléchargée.


 	sudo docker run -d --rm --volume="./volume_tp:/home/" -p 220:22 waelbachta/ubuntu_ssh_labs:latest

Cela vous permet de lancer le conteneur qui établit une connexion ssh sur le port 220 avec votre machine.

- Se connecter avec un ssh X à votre conteneur en écrivant :
     ```
    ssh -X -p 220 root@localhost
 	```
 Vous aurez à saisir le mot de passe du conteneur qui est root123

 - Aller dans /home et vérifier qu'il correspond bien à volume\_tp.

 - Accéder au répertoire qui correspond à votre TP.

 - Vous pouvez ainsi modifier le code sur votre ordinateur personnel et et l'interpréter sur le conteneur.

