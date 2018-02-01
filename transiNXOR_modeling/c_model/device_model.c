#ifdef OSX_ACCELERATE
#  include <Accelerate/Accelerate.h>
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#  include <mkl_cblas.h>
#elif defined(__GNUC__) || defined(__GNUG__)
#  include <cblas.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* -------------------- PARAMETERS -------------------- */
float tanh_fc_layer_0_w[] = { 0.60636699,-0.7843641 , 0.85639101, 0.62267089, 0.65765762,-0.85957742,
 -0.93166775,-0.60751784, 0.80550343, 0.7906968 , 0.84626752,-0.26057673,
  0.68305463, 0.68828636,-0.61112815,-0.62441045};
float sig_fc_layer_0_w[] = { 3.32971644, 2.67833281, 3.51681685, 3.48378539,-3.43650889,-3.42502928,
 -3.11738086,-3.13322043,-2.76904488,-2.70317841,-2.5338881 ,-2.48465824,
 -2.80293179,-2.86709142, 2.44205284, 2.54085016,-4.42845535,-4.3377223 ,
 -6.57501364,-6.39385033,-2.71946144,-2.92146325,-4.62975693,-4.66682625,
 -2.83418751,-2.60433578, 2.91891313, 3.07859969,-2.77421951,-2.92486024,
 -6.48648977,-6.56851912};
float sig_fc_layer_0_b[] = { 3.26407242, 4.81284904,-0.68809128,-0.63646072, 1.5151161 ,-0.38658351,
  1.48387098, 1.86894655,-0.76280105,-0.25678775, 0.7160567 , 0.00834461,
  0.44542199, 1.57467723, 1.22834623, 0.88106745};
float inter_embed_layer_0_w[] = {-0.13356361,-0.20037866,-0.09189396,-0.01600051,-0.02312116, 0.19110282,
  0.02952356,-0.07895931, 0.12583067, 0.17326447,-0.09066953,-0.02792229,
  0.06414478, 0.14777824,-0.32540792, 0.03747429,-0.88580412, 0.82912654,
 -0.48206887,-0.7357409 ,-0.59194219, 0.65027052, 0.60921484, 0.44987446,
 -0.80860698,-0.87619007,-0.84142578, 0.75062329,-0.54606414,-0.88616443,
  0.4907532 , 0.82960874, 0.29425865,-0.26536852, 0.47873792, 0.57867932,
  0.4237352 ,-0.55406177,-0.600128  ,-0.53420633, 0.31491429, 0.64536148,
  0.52712333,-0.13624769, 0.58888054, 0.63483357,-0.49921766,-0.30397704,
  0.40894163,-0.38360476, 0.39920059, 0.69924247, 0.76900345,-0.65872157,
 -0.59825772,-0.77880216, 0.4359498 , 0.38766301, 0.57630903,-0.35008252,
  0.51927638, 0.87368512,-0.58807701,-0.78674561, 0.20076276,-0.67299223,
  0.47415611, 0.2369862 , 0.29675221,-0.4947497 ,-0.49157164,-0.61125124,
  0.47856888, 0.3972626 , 0.67190635,-0.16027014, 0.23633187, 0.41481942,
 -0.24661447,-0.37899026, 0.73957157,-0.73716414, 0.39944762, 0.32685801,
  0.74763709,-0.54096258,-0.70510501,-0.69544727, 0.46589476, 0.30328941,
  0.50679618,-0.54770428, 0.67714798, 0.31840953,-0.52646762,-0.6734665 ,
  0.66403472,-0.50929111, 0.60949445, 0.29894948, 0.54832876,-0.43043754,
 -0.62806648,-0.32260811, 0.27299577, 0.22928992, 0.41735476,-0.41659391,
  0.58900058, 0.34956434,-0.51149166,-0.23120254, 0.09644908, 0.31813723,
  0.0086517 ,-0.1388271 , 0.13027129, 0.24300961, 0.20221218, 0.20602183,
 -0.26366279, 0.14586274,-0.16121301, 0.3235034 , 0.18075168,-0.01707607,
 -0.04541631, 0.11869007, 0.51609403,-0.67859095, 0.32215124, 0.59517097,
  0.39564273,-0.50737268,-0.37011206,-0.29165667, 0.47533107, 0.41612324,
  0.32274371,-0.57235265, 0.42941418, 0.30779219,-0.39233992,-0.39098379,
  0.17753035,-0.05778405, 0.30305099, 0.45001045, 0.31906927,-0.44594762,
 -0.18880704,-0.20779689, 0.50140786, 0.05415041, 0.43749252,-0.40820345,
  0.12001662, 0.33024663,-0.07922753,-0.2134724 , 0.68704391,-0.26709166,
  0.66386968, 0.47670069, 0.40170354,-0.57388288,-0.74727839,-0.22838141,
  0.27290419, 0.58162683, 0.6217469 ,-0.54196411, 0.70472395, 0.37977585,
 -0.32092211,-0.23424459, 0.09361789,-0.1747648 , 0.07389509, 0.29098067,
  0.34178025,-0.20229654,-0.34100896,-0.02446492, 0.0768984 , 0.17737591,
  0.42167312,-0.0293482 , 0.11193332, 0.16135123, 0.0099111 ,-0.24955417,
  0.38476878,-0.48188168, 0.44268334, 0.5891121 , 0.47850093,-0.57630992,
 -0.52598602,-0.50936908, 0.27590799, 0.46111074, 0.59856892,-0.28567937,
  0.56465524, 0.36137426,-0.17156564,-0.49813148,-0.25067973, 0.27211794,
 -0.19638547,-0.36409348,-0.54438251, 0.27661446, 0.55109513, 0.55915868,
 -0.37540019,-0.35113055,-0.28912622,-0.00302252,-0.39936376,-0.28248474,
  0.28198892, 0.15249993, 0.48248598,-0.33658993, 0.72223783, 0.32164937,
  0.642528  ,-0.26330224,-0.54696929,-0.30164829, 0.37418836, 0.54195547,
  0.48575681,-0.50256956, 0.49584359, 0.67499286,-0.69925761,-0.23482668,
 -0.06477819, 0.0750494 ,-0.11402509,-0.1075597 ,-0.06076488,-0.04802775,
  0.06608023,-0.27598518, 0.05242253, 0.23915355, 0.12514596, 0.09017728,
  0.11401875,-0.29996902, 0.48432577,-0.18775249};
float inter_embed_layer_0_b[] = { 3.10233831, 4.65334797,-0.48731178,-0.32086244, 1.40597188, 0.28939408,
  1.64004397, 2.36225653,-0.40210447,-0.25682715, 0.55496764,-0.39920989,
  1.19127333, 0.69073498, 0.81850708, 1.18893385};
float tanh_fc_layer_1_w[] = {-0.08827368,-0.01890268, 0.07143544, 0.05593424,-0.42075509, 0.05182291,
  0.07634114, 0.31350958,-0.04693743,-0.20171477,-0.11293143, 0.17040651,
 -0.40733641,-0.29139894, 0.30449891,-0.03379741, 0.17359017,-0.0480516 ,
  0.05840205,-0.13854644, 0.19120702, 0.15141873, 0.14343266, 0.10843758,
  0.04272978,-0.01157918,-0.00257413,-0.16868931, 0.26058525, 0.0494815 ,
 -0.1586732 ,-0.12847811, 0.05972605, 0.11108477,-0.09765945,-0.19987361,
 -0.00918475, 0.30382064, 0.09734508, 0.07118802,-0.20234394,-0.31669942,
 -0.19978091,-0.02677872,-0.01330673,-0.4209891 , 0.42798969,-0.07283535,
 -0.15684804,-0.24166083,-0.02088156,-0.03284889,-0.17361839,-0.24459977,
  0.0561454 ,-0.23396674, 0.26182428,-0.17191802,-0.14067739, 0.07590986,
  0.15353291,-0.11899091,-0.18624498,-0.13449636, 0.09200286, 0.15813749,
 -0.0492476 ,-0.20272036,-0.14676508, 0.26555604, 0.26156679, 0.12295433,
  0.01043294,-0.18917279, 0.06176221, 0.07612546, 0.18133166,-0.00968119,
 -0.12311164,-0.0293569 , 0.16372342,-0.25920862, 0.30556098, 0.01940914,
  0.03345706,-0.0848458 ,-0.39745495,-0.21142621, 0.12450082, 0.16650432,
  0.30117637, 0.06326885, 0.29621443, 0.02022299,-0.36970326,-0.02686415,
  0.05885321,-0.24545993, 0.19448932,-0.00367196, 0.29998934,-0.00337604,
 -0.20087063,-0.02214952, 0.05879935, 0.31439018, 0.0564672 ,-0.22037122,
  0.22480166, 0.17895086, 0.10899052,-0.28281474, 0.19013387,-0.21628307,
  0.27202699, 0.39503956, 0.43147948,-0.01168653,-0.10396758,-0.27647668,
  0.31416404, 0.458381  , 0.17135024,-0.29747137, 0.46503291, 0.23784806,
 -0.26657915,-0.08643638,-0.65194035, 0.7278235 ,-0.26614434,-0.6878857 ,
 -0.53113449, 0.53948689, 0.15021864, 0.79592663,-0.33547372,-0.20779122,
 -0.2447302 , 0.5379321 ,-0.77723163,-0.37213743, 0.68799251, 0.68343204,
  0.32736948,-0.19724753,-0.06851855, 0.14556748, 0.33767995,-0.16229306,
 -0.14721644,-0.28513682, 0.32058629,-0.10148175, 0.13142318,-0.01313008,
 -0.00597564, 0.25098169,-0.33346796,-0.12739348, 0.16032813,-0.05507826,
  0.32156947, 0.35023415, 0.48007789,-0.02525032,-0.17239647,-0.45250648,
 -0.03900697, 0.34205362, 0.20281361,-0.20780842, 0.34105512, 0.35704154,
 -0.15773265,-0.03404807,-0.44491386, 0.64613223,-0.32128981,-0.35208344,
 -0.39323521, 0.48505312, 0.14202201, 0.41586685,-0.30926287,-0.4809323 ,
 -0.67204791, 0.52408463,-0.78173816,-0.68536162, 0.46986756, 0.61277109,
  0.09380364,-0.03672266, 0.15840486, 0.06959576, 0.20246397, 0.05978293,
 -0.20800707,-0.31356096,-0.04822062, 0.21963492, 0.30873585, 0.05048867,
  0.08104847, 0.04496771,-0.2296555 ,-0.04404226, 0.008748  , 0.03928152,
 -0.00306445,-0.32951632, 0.03173273, 0.08718503,-0.04347861, 0.0128813 ,
 -0.33970141,-0.14854717,-0.28740153, 0.18288764,-0.08999074,-0.27711982,
 -0.02734668, 0.31145227,-0.00146304, 0.04895189,-0.16702886,-0.17964585,
 -0.19237877, 0.29643995, 0.18995114,-0.05493777,-0.01749374,-0.08303705,
  0.08703379,-0.09247132, 0.03933056,-0.19192471, 0.01224054,-0.11767911,
  0.35220513, 0.06534898, 0.21924464, 0.21777825,-0.08050006, 0.06272022,
  0.09283616,-0.34372216, 0.33028764,-0.10935724,-0.03046845, 0.07438781,
  0.36216411,-0.04107988, 0.00450453, 0.01886649};
float sig_fc_layer_1_w[] = {  1.68290031e+00,  1.29600680e+00,  6.10128880e-01,  9.57739413e-01,
  -2.86406628e-03,  7.74395227e-01,  2.23349303e-01,  1.72151446e+00,
   9.17960167e-01,  1.18947947e+00,  6.59660101e-01,  9.57283318e-01,
   3.59467596e-01, -1.26965630e+00, -2.48148330e-02,  1.06432247e+00,
   7.73811817e-01,  1.01672328e+00,  1.62544131e+00,  1.59139013e+00,
  -1.43237516e-01,  9.81633723e-01,  2.97195852e-01,  3.48846865e+00,
   2.03685141e+00,  2.32601905e+00,  8.24407756e-01,  1.71354830e+00,
   9.93549168e-01, -2.95872784e+00,  3.78349453e-01,  2.70754647e+00,
  -5.92645550e+00, -2.96016741e+00, -1.42606044e+00, -1.54785657e+00,
  -1.51380062e+00, -2.31840801e+00, -1.70260978e+00, -3.57509637e+00,
  -5.90871274e-01, -8.75029266e-01, -1.85707390e+00, -5.71329296e-01,
  -1.70197046e+00, -5.06949499e-02, -1.69047546e+00, -1.05023909e+00,
  -1.38760650e+00, -4.83539164e-01,  2.20693946e+00,  1.68878174e+00,
   9.10619378e-01,  8.40503216e-01,  1.36961043e+00, -1.16613112e-01,
   3.15230894e+00,  3.78858423e+00,  6.61217630e-01,  2.81279707e+00,
   1.07243621e+00, -2.00746036e+00,  5.88150382e-01,  3.97624326e+00,
   6.51588142e-01,  3.76751542e-01, -7.42036283e-01, -8.37909400e-01,
  -3.13572198e-01, -3.80456656e-01, -3.61654967e-01,  1.58330500e-01,
  -9.14915025e-01, -6.21763051e-01, -7.27410972e-01, -8.27076375e-01,
  -8.08867931e-01,  6.21391118e-01, -1.92749217e-01, -1.03937292e+00,
  -2.75122136e-01,  2.42106438e-01,  7.84052312e-01,  3.03979248e-01,
   1.02381790e+00,  6.15216851e-01,  1.36146307e+00, -1.08669795e-01,
   4.15959954e-01,  5.46801925e-01,  5.26421785e-01,  6.54328585e-01,
   7.26860762e-01, -3.95150900e-01,  3.73074055e-01,  6.82670414e-01,
  -3.90848547e-01, -4.45651412e-01,  8.31683874e-01,  5.94954133e-01,
   4.81224686e-01,  3.93763214e-01,  8.21283281e-01, -3.07681024e-01,
   7.33263612e-01,  7.78450906e-01,  4.96040076e-01,  2.37435326e-01,
   6.39899075e-01, -7.09069967e-01,  4.47156012e-01,  8.46663356e-01,
   2.13048950e-01,  4.16728318e-01,  4.15211916e-01,  4.27574128e-01,
   1.93145204e+00,  8.78356457e-01,  2.47277164e+00,  6.80608451e-01,
   3.91173482e-01,  7.97024906e-01,  9.36025977e-01,  3.83622885e-01,
   1.39810002e+00, -7.86348879e-02,  1.18869579e+00,  4.25559342e-01,
  -1.58042741e+00, -5.86227298e-01,  1.53797793e+00,  1.77489197e+00,
   1.05741060e+00,  7.07144022e-01,  1.35285270e+00,  4.97437678e-02,
   2.09190702e+00,  2.62553763e+00,  6.84470475e-01,  1.50431919e+00,
   8.19585800e-01, -1.12903881e+00,  8.12881410e-01,  2.46386647e+00,
  -6.15425050e-01,  7.90612623e-02,  6.60432577e-01,  5.77456415e-01,
   5.81517339e-01,  6.92960441e-01,  5.51255345e-01, -3.22389424e-01,
   8.12639058e-01,  6.81083500e-01,  4.51712430e-01,  5.04196346e-01,
   6.20364249e-01, -3.95496011e-01,  1.64877295e-01,  6.87069535e-01,
   6.44562185e-01,  1.42769098e-01, -9.02340114e-01, -8.79349232e-01,
  -2.57825524e-01, -5.01106858e-01, -5.33576608e-01, -7.94390589e-02,
  -9.86071825e-01, -9.28971946e-01, -3.39186996e-01, -6.30601466e-01,
  -7.06723928e-01,  7.68884480e-01, -5.01356900e-01, -6.49438500e-01,
  -7.64071822e-01, -3.82169813e-01,  1.06690288e+00,  9.67268169e-01,
   5.00058591e-01,  5.08524776e-01,  9.35863197e-01, -1.31483868e-01,
   7.27652967e-01,  9.64440703e-01,  8.10177565e-01,  6.57572210e-01,
   9.60966825e-01, -4.82637107e-01,  5.32948792e-01,  8.30479920e-01,
  -3.53080320e+00, -9.47089767e+00, -4.02425408e-01, -3.49615574e-01,
  -2.12556839e+00, -1.62540221e+00, -2.17672420e+00, -3.88938713e+00,
   5.74321449e-01,  4.03028071e-01, -2.23946953e+00,  9.52517509e-01,
  -2.32807374e+00, -2.58910227e+00, -1.92154121e+00,  2.75930732e-01,
  -2.42943168e+00, -4.34267950e+00, -5.39835095e-01, -4.38055664e-01,
  -1.15326774e+00, -1.29757392e+00, -7.21484125e-01, -2.48538804e+00,
   2.66841024e-01,  3.48553896e-01, -1.07111323e+00,  5.51379561e-01,
  -8.09254229e-01, -1.01940107e+00, -4.66329545e-01, -7.65344799e-02,
   1.57902288e+00,  2.56823826e+00,  1.30987668e+00,  8.63732159e-01,
   9.49105859e-01,  2.60081142e-01,  1.10688496e+00,  2.19606233e+00,
   5.73126435e-01,  5.59287071e-01,  1.32808864e-01,  2.26730540e-01,
   3.85817617e-01, -4.86099511e-01,  4.41579849e-01,  2.97422051e-01,
  -1.91661084e+00, -1.35386646e+00,  3.68874133e-01,  4.07954365e-01,
  -5.63918948e-01, -4.68191691e-02, -1.62710901e-02, -7.73199558e-01,
   2.96359301e-01,  1.05017805e+00, -2.02148601e-01,  5.28720021e-01,
  -2.62862325e-01, -1.39083850e+00, -6.16562068e-01,  1.05514777e+00};
float sig_fc_layer_1_b[] = {-0.33415931,-1.79970944,-0.17109834,-0.32592073, 0.26304644,-0.098612  ,
 -0.14835146, 0.38736123,-0.32605743,-0.40711233, 0.30397499,-0.08808478,
 -0.86780018, 0.00540383,-0.20601793,-0.81818134};
float inter_embed_layer_1_w[] = { 0.00514287,-0.03425698, 0.23224388,-0.16783664, 0.05019943,-0.00370265,
 -0.09616288,-0.21054579, 0.72996467,-0.05976339,-0.48845097, 0.35790461,
 -0.06983863, 0.12699544,-0.01959171,-0.37604493,-0.10798005, 0.22863549,
  0.06502949,-0.13589658, 0.41101202,-0.22640683,-0.42025191,-0.63093549,
  0.79128313,-0.30772924,-0.37560645, 0.88900721,-0.24367149, 0.07467505,
  0.01844595, 0.11757893,-0.97082424, 0.73570246,-0.8513937 , 0.1698419 ,
 -0.1190939 , 0.59330589, 0.62258142, 0.86776686,-0.9653213 , 0.93742949,
  0.63230032,-0.87283808, 0.86531228,-0.75794488,-0.43088004, 0.7972666 ,
  0.13910648, 0.01281421, 0.26906532,-0.01087992, 0.55608761,-0.44806802,
 -0.26020077,-0.62344456, 0.81148911,-0.27644688,-0.64969271, 0.711514  ,
 -0.33892563, 0.06765188, 0.45256588, 0.17218606,-0.02455966, 0.28086036,
 -0.07852843, 0.27366877,-0.17359589,-0.05766181,-0.1517417 ,-0.12899028,
 -0.37950128, 0.02837986,-0.07702877,-0.09133635,-0.04719894, 0.09361303,
 -0.06049229, 0.07035531, 0.14453518,-0.09355272,-0.35477069, 0.07816452,
  0.07650713, 0.26781258, 0.32599607, 0.0957555 , 0.05914896,-0.01853588,
 -0.07232285,-0.08363721,-0.03562548, 0.16731305,-0.15733863, 0.15751263,
  0.13988498,-0.18303922, 0.20803702,-0.25405413,-0.01522241,-0.12704214,
  0.28206831, 0.0676183 , 0.12074625,-0.28472596, 0.07675024,-0.11660381,
 -0.2480012 , 0.21177675, 0.1601906 ,-0.22107179, 0.15835656,-0.13928752,
  0.27280325,-0.2678968 ,-0.11548413,-0.00463345,-0.05196566,-0.36956525,
  0.66444647,-0.18553515,-0.16017401, 0.39849761, 0.1113017 , 0.23346302,
 -0.04288021,-0.41470233, 0.01632302, 0.12093561, 0.13930428,-0.27821499,
  0.13389082,-0.22089589,-0.50459588,-0.1670372 , 0.36345208,-0.11880165,
 -0.52265847, 0.66842359,-0.30750313, 0.23840201, 0.06293114, 0.07245699,
  0.10856903,-0.12012406, 0.11750332, 0.02712692,-0.07305774, 0.04790964,
 -0.05569272, 0.12750395, 0.40940559, 0.03316095, 0.24436542,-0.17158468,
  0.19972356, 0.05262632,-0.24837589,-0.15339708,-0.27986813, 0.19565831,
 -0.0616548 , 0.34358379, 0.07387578,-0.0310613 ,-0.18774712, 0.04921414,
 -0.29589009, 0.14633447,-0.01465159, 0.02970632,-0.064417  ,-0.33679911,
  0.05326089,-0.01144587, 0.02454002,-0.06820385,-0.08370374,-0.35281119,
  0.21014765,-0.11848526,-0.05445666,-0.29504308, 0.19913048,-0.1876355 ,
 -0.10760485, 0.03858972, 0.0768126 , 0.3488701 ,-0.14851637, 0.03440852,
 -0.68202066, 0.27568808,-0.93546999, 0.5154925 ,-0.34074342, 0.8789379 ,
  0.427486  , 1.09244871,-1.33436286, 1.11686277, 0.79961205,-1.19168317,
  0.71315205,-1.02392447,-0.14616872, 0.26927909, 0.02267235, 0.2409555 ,
 -0.18452798,-0.15945378, 0.15276043, 0.33753273,-0.25683841, 0.31520689,
 -0.75716323, 0.31177881, 0.49852216,-0.74061698,-0.0908578 ,-0.37514016,
  0.25320572, 0.09673104, 0.26554292, 0.02742671, 0.19289577,-0.16179237,
 -0.03223583,-0.22221477, 0.18226218,-0.36966965, 1.05075288,-0.15543467,
 -0.16096455, 0.48487732,-0.1058187 , 0.18575069,-0.14910188,-0.38402629,
 -0.06705336,-0.12657902,-0.37341878,-0.06995878,-0.16802715, 0.03804735,
  0.11127888, 0.07744842,-0.28410652,-0.0585089 , 0.50098246,-0.64215404,
  0.00393264,-0.11909654,-0.05065621, 0.30556625};
float inter_embed_layer_1_b[] = {-0.60707593,-2.04309988,-0.31986776,-0.21636951, 0.02366933,-0.25027293,
 -0.33114165, 0.27665567,-0.38065922,-0.3331416 , 0.10468668,-0.05313139,
 -0.63926023, 0.00703273,-0.12343286,-0.87790304};
float tanh_fc_layer_2_w[] = {-0.10009687,-0.05664347,-0.26263443,-0.05287291, 0.02603611, 0.46590528,
 -0.15352736, 0.55736458,-0.26080704, 0.38651729, 0.55245847,-0.70078772,
  0.30649221,-0.23919043,-0.04619053, 0.04800953};
float sig_fc_layer_2_w[] = {-1.22382748,-1.17117012, 2.62274075,-0.93625355, 0.79836988,-0.90296066,
 -0.81702638,-0.84620887,-0.88588923,-0.77635616, 0.81106114,-0.62501693,
  4.11785984, 2.363729  ,-1.2948724 ,-0.83102828};
float sig_fc_layer_2_b[] = { 0.1601387};
float inter_embed_layer_2_w[] = { 0.67124748};
float inter_embed_layer_2_b[] = {-0.33187959};

/* -------------------- HELPER FUNCTIONS -------------------- */
void fc(const int m, const int n, const int k, 
	float *W, float *I, float *B) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, 1.0, W, k, I, n, 1.0, B, n);
}

void matmul(const int m, const int n, const int k, 
	float *W, float *I, float *O) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, 1.0, W, k, I, n, 0.0, O, n);
}

void add(const int n, float *a, float *b) {
	cblas_saxpy(n, 1.0, a, 1, b, 1);
}

void sig_act(float *a, const int len) {
	int i;
	for (i=0; i < len; i++)
		a[i] = (tanh(a[i]/2) + 1)/2;
}

void tanh_act(float *a, const int len) {
	int i;
	for (i=0; i < len; i++) {
		a[i] = tanh(a[i]);
	}
}

void print_array(float *a, const int len) {
	int i;
	for (i=0; i < len; i++)
		printf("%f, ", a[i]);
	printf("\n");
}

/* -------------------- DEVICE MODEL -------------------- */
float device_model( 
	const float vtg, 
	const float vbg, 
	const float vds,
	const float w
) {
	float vg[2] = {(vtg-0.1)/0.1, (vbg-0.1)/0.1};
	float vd[1] = {vds/0.2};
	float tanh_temp0[16] = {0};
	float tanh_temp1[16] = {0};
	// Layer 0
  fc(16, 1, 2, sig_fc_layer_0_w, vg, sig_fc_layer_0_b);
	matmul(16, 1, 1, tanh_fc_layer_0_w, vd, tanh_temp0);
	fc(16, 1, 16, inter_embed_layer_0_w, tanh_temp0, inter_embed_layer_0_b);
	add(16, inter_embed_layer_0_b, sig_fc_layer_0_b);
  sig_act(sig_fc_layer_0_b, 16);
  tanh_act(tanh_temp0, 16);

	// Layer 1
  fc(16, 1, 16, sig_fc_layer_1_w, sig_fc_layer_0_b, sig_fc_layer_1_b);
	matmul(16, 1, 16, tanh_fc_layer_1_w, tanh_temp0, tanh_temp1);
	fc(16, 1, 16, inter_embed_layer_1_w, tanh_temp0, inter_embed_layer_1_b);
	add(16, inter_embed_layer_1_b, sig_fc_layer_1_b);
  print_array(sig_fc_layer_1_b, 16);
  sig_act(sig_fc_layer_1_b, 16);
  tanh_act(tanh_temp1, 16);
	
  // print_array(tanh_temp1, 16);
	// Layer 2
  fc(1, 1, 16, sig_fc_layer_2_w, sig_fc_layer_1_b, sig_fc_layer_2_b);
	matmul(1, 1, 16, tanh_fc_layer_2_w, tanh_temp1, tanh_temp0);
	fc(1, 1, 1, inter_embed_layer_2_w, tanh_temp0, inter_embed_layer_2_b);
	add(1, inter_embed_layer_2_b, sig_fc_layer_2_b);
  tanh_act(tanh_temp0, 1);
	sig_act(sig_fc_layer_2_b, 1);
	// Output	
	return sig_fc_layer_2_b[0] * tanh_temp0[0] * 53.65093994 * w;
}

int main(int argc, char** argv) {
	float id = device_model(0.2, 0.2, 0.2, 1);
	// float a[] = {1, 2, 3, 4};
	// float b[] = {1, 2};
	// float c[] = {1, 1};
	// matmul(2, 1, 2, a, c, b);
	// print_array(c, 2);
	// print_array(b, 2);
	// add(2, b, c);
	// print_array(c, 2);
	printf("%f\n", id);
}