% Simulation of Spring Pendulum
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Warning : On running this the workspace memory will be deleted. Save if
% any data present before running the code !!
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%-------------------------------------------------------------------------
% Original code written by:                                               |
%       Siva Srinivas Kolukula      (e-mail : allwayzitzme@gmail.com)     |
%          http://sites.google.com/site/kolukulasivasrinivas/             |
%-------------------------------------------------------------------------
% Modified by:                                                            |
%       Pau Urdetix Diaz            (e-mail : purdeitx@unizar.com)        |
%          https://amb.unizar.es/people/pau-urdeitx/                      |
%          github: https://github.com/a-tierz/TINN_tutorial               |
% Modifications: 
%   - Add dissipative model
%   - Save state vector -> database generation
%-------------------------------------------------------------------------
clear ;clc ;
% Properties of Pendulum (Can be altered)
grav = true ;               % false if only want elastic potential energy
M = 2 ;                     % Mass of the pendulum
L = 1 ;                     % Length of the Pendulum
K = 5 ;                     % Spring Constant
d = 0.00 ;                  % Damping coefficient 

if grav == true
    g = 9.81 ;              % Acceleration due to gravity
else
    g = 0.00 ;              % Acceleration due to gravity
end

% Time configuration
duration = 20;              % Duration of the Simulation 
fps = 60;                   % Frames per second

%movie = true;              % true if wanted to save animation as avi file
movie = false ;             % false if only want to view animation
arrow = true ;              % Shows the direction of phase plane plot
%arrow = false ;            % Will not show the direction of phase plane plot
interval = [0, duration];                  % Time span

% In case you need to save a database with N different initial conditions
numberOfExperiments = 1;

% Initial Boundary Conditions (Can be altered)
r_rol = [1.5, 2.0, 2.5] ;   % Extension Length
rdot = 1. ;                 % Radial velocity
angle_rol = [20, 30, 40];   % Initial angle (degres)  
Phidot = 0.1;               % Angular velocity

for i=1:numberOfExperiments
    r=r_rol(randi(length(r_rol)));
    angle=angle_rol(randi(length(angle_rol)));
    Phi=angle*pi/180;
    ivp=[r ;rdot ;Phi ;Phidot ;g ;M ;L ; K; d];   % Initial value's for the problem
    % Simulation of Simple Pendulum
    [Z, dt]  = Animation(ivp,duration,fps,movie,arrow);
    if (abs(d)>0)
        % Dissipative pendulum
        name = strcat('data_r',sprintf('%.1f',r),'_theta',sprintf('%.0f',angle),'_dis.mat')
        save(name,"Z","duration","fps","dt","-mat");
    else
        % Reversible pendulum 
        name = strcat('data_r',sprintf('%.1f',r),'_theta',sprintf('%.0f',angle),'_rev.mat')
        save(name,"Z","duration","fps","dt","-mat");
    end
end

