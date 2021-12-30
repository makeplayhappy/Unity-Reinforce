using UnityEngine;

/**
   * This is reasonably redundant in C#... better to write getters and setter in the inheritted classes
   * keeping here for clarity for anyone debugging from the original source
   * 
   * Get property value of Opt by fieldname
   * @param fieldname name of the property as `string`
   * @returns value or `undefined` (string) if no value exists
   * /
  public get(fieldname:string): any | undefined {
    return this[fieldname] ? this[fieldname] : undefined;
  }
** * */

namespace Reinforce{
    public class Options {



    }
}
