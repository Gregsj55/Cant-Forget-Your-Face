# CSCI-3308-Fall21-011-08
Can’t Forget Your Face

Heroku Link: https://cant-forget-your-face.herokuapp.com

Link to Project Board: https://csci-3308-fall21-010-08.atlassian.net/jira/software/projects/C800/boards/1

Link to roadmap: https://csci-3308-fall21-010-08.atlassian.net/jira/software/projects/C800/boards/1/roadmap

Link to video: https://www.youtube.com/watch?v=KBbrVDPI-4Y&t=2s

Application Description: 
Can’t forget your face is a website that when visited will prompt the user to enter a username and will train CV software to recognise their face. After training, the program recognizes that person's face, and will allow access to a blank text document where the user can write whatever they want. The facial recognition data, the username, and the document are  persisted, so a user can log into their document at any time with their face. The practical application for this program is to allow users to keep their information in a safe and a secure spot where they can access it whenever they would like.
We were inspired to make this application because we found that regular two factor authentication (TFA) was too slow. When you would enable TFA on a website you could often wait up to an hour to receive a text message or email confirming your account. Our solution is to allow a user to quickly and easily log into their account using their face, which you can never forget! 
We decided to use this feature to let a user access a secure text document where they could store any information desired. However, that does not stop us from applying our solution to TFA in other places.

How to run it:
Our app was deployed using docker. When first pulling the repository one must first do a docker-compose build command and wait for that to finish. It should take about a minute. After that simply run docker-compose up and the application will be live. It can be accessed on localhost:5000 through Chrome.

Some special details:
When running this application will create new folders inthe repository to localy store images. Please  dont commit these to git or else it causes privacy concerns. Also, both the username and the password  must be unique. If you are running into sqlalchemy error pages yourusername/password is probably already in the database and you must select a new one.
When using the facial recognition try to have the best lighting possible. If your lighting is poor you will have worse results.
