using UnityEngine;

namespace Reinforce{
    public class Environment {
        public readonly int width;
        public readonly int height;
        public readonly int numberOfStates;
        public readonly int numberOfActions;


        public Environment(int w, int h, int s, int a){
            width = w;
            height = h;
            numberOfStates = s;
            numberOfActions = a;
        }

    }
}
